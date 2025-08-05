"""
ops0 Pandas Integration

Data processing with pandas - automatic serialization to Parquet,
data validation, and memory optimization.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import io

from .base import (
    BaseIntegration,
    DatasetWrapper,
    IntegrationCapability,
    SerializationHandler,
    DatasetMetadata,
)

logger = logging.getLogger(__name__)


class PandasSerializer(SerializationHandler):
    """Pandas DataFrame serialization handler"""

    def __init__(self, format: str = 'parquet'):
        """
        Initialize serializer.

        Args:
            format: Serialization format ('parquet', 'feather', 'pickle', 'csv')
        """
        self.format = format

    def serialize(self, obj: Any) -> bytes:
        """Serialize pandas DataFrame"""
        import pandas as pd

        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(obj)}")

        buffer = io.BytesIO()

        if self.format == 'parquet':
            obj.to_parquet(buffer, engine='pyarrow', compression='snappy')
        elif self.format == 'feather':
            obj.to_feather(buffer)
        elif self.format == 'pickle':
            obj.to_pickle(buffer, protocol=5)
        elif self.format == 'csv':
            obj.to_csv(buffer, index=False)
        else:
            raise ValueError(f"Unknown format: {self.format}")

        return buffer.getvalue()

    def deserialize(self, data: bytes) -> Any:
        """Deserialize to pandas DataFrame"""
        import pandas as pd

        buffer = io.BytesIO(data)

        if self.format == 'parquet':
            return pd.read_parquet(buffer, engine='pyarrow')
        elif self.format == 'feather':
            return pd.read_feather(buffer)
        elif self.format == 'pickle':
            return pd.read_pickle(buffer)
        elif self.format == 'csv':
            return pd.read_csv(buffer)
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def get_format(self) -> str:
        return f"pandas-{self.format}"


class PandasDatasetWrapper(DatasetWrapper):
    """Wrapper for pandas DataFrames"""

    def _extract_metadata(self) -> DatasetMetadata:
        """Extract metadata from pandas DataFrame"""
        import pandas as pd

        if not isinstance(self.data, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(self.data)}")

        # Basic info
        shape = self.data.shape
        memory_usage = self.data.memory_usage(deep=True).sum()

        # Data types
        dtypes = {col: str(dtype) for col, dtype in self.data.dtypes.items()}

        # Column stats
        stats = {}

        # Numeric columns
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats['numeric'] = {
                col: {
                    'mean': float(self.data[col].mean()),
                    'std': float(self.data[col].std()),
                    'min': float(self.data[col].min()),
                    'max': float(self.data[col].max()),
                    'nulls': int(self.data[col].isnull().sum()),
                }
                for col in numeric_cols
            }

        # Categorical columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            stats['categorical'] = {
                col: {
                    'unique': int(self.data[col].nunique()),
                    'nulls': int(self.data[col].isnull().sum()),
                    'top_values': self.data[col].value_counts().head(5).to_dict(),
                }
                for col in categorical_cols
            }

        return DatasetMetadata(
            format='pandas.DataFrame',
            shape=shape,
            dtypes=dtypes,
            size_bytes=memory_usage,
            n_samples=shape[0],
            n_features=shape[1],
            feature_columns=list(self.data.columns),
            stats=stats,
        )

    def to_numpy(self) -> Any:
        """Convert to numpy array"""
        return self.data.values

    def split(self, train_size: float = 0.8) -> Tuple[Any, Any]:
        """Split into train/test sets"""
        split_idx = int(len(self.data) * train_size)

        # Maintain DataFrame type
        train_df = self.data.iloc[:split_idx].copy()
        test_df = self.data.iloc[split_idx:].copy()

        return train_df, test_df

    def sample(self, n: int, random_state: int = None) -> Any:
        """Sample n rows from DataFrame"""
        return self.data.sample(n=min(n, len(self.data)), random_state=random_state)


class PandasIntegration(BaseIntegration):
    """Integration for pandas data processing"""

    def __init__(self):
        super().__init__()
        self._default_format = 'parquet'

    @property
    def is_available(self) -> bool:
        """Check if pandas is installed"""
        try:
            import pandas as pd
            return True
        except ImportError:
            return False

    def _define_capabilities(self) -> List[IntegrationCapability]:
        """Define pandas capabilities"""
        return [
            IntegrationCapability.SERIALIZATION,
            IntegrationCapability.DATA_VALIDATION,
            IntegrationCapability.VISUALIZATION,
        ]

    def get_version(self) -> str:
        """Get pandas version"""
        try:
            import pandas as pd
            return pd.__version__
        except ImportError:
            return "not installed"

    def wrap_model(self, model: Any) -> None:
        """Pandas doesn't have models"""
        raise NotImplementedError("Pandas integration doesn't support models")

    def wrap_dataset(self, data: Any) -> DatasetWrapper:
        """Wrap pandas DataFrame"""
        return PandasDatasetWrapper(data)

    def get_serialization_handler(self) -> SerializationHandler:
        """Get pandas serialization handler"""
        if self._serialization_handler is None:
            self._serialization_handler = PandasSerializer(format=self._default_format)
        return self._serialization_handler

    def optimize_dataframe(self, df: Any, **kwargs) -> Any:
        """Optimize DataFrame memory usage"""
        import pandas as pd
        import numpy as np

        if not isinstance(df, pd.DataFrame):
            return df

        # Original memory usage
        start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
        logger.info(f"Memory usage before optimization: {start_mem:.2f} MB")

        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

        # Optimize float columns
        for col in df.select_dtypes(include=['float']).columns:
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

        # Convert object columns to category if appropriate
        for col in df.select_dtypes(include=['object']).columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])

            if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')

        # Final memory usage
        end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
        logger.info(f"Memory usage after optimization: {end_mem:.2f} MB")
        logger.info(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

        return df

    def validate_dataframe(self, df: Any, schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate DataFrame against schema"""
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            return {'valid': False, 'errors': ['Not a pandas DataFrame']}

        errors = []
        warnings = []

        # Check for nulls
        null_counts = df.isnull().sum()
        if null_counts.any():
            null_cols = null_counts[null_counts > 0]
            warnings.append(f"Columns with nulls: {null_cols.to_dict()}")

        # Check for duplicates
        if df.duplicated().any():
            warnings.append(f"Found {df.duplicated().sum()} duplicate rows")

        # Schema validation if provided
        if schema:
            # Check required columns
            if 'required_columns' in schema:
                missing = set(schema['required_columns']) - set(df.columns)
                if missing:
                    errors.append(f"Missing required columns: {missing}")

            # Check data types
            if 'dtypes' in schema:
                for col, expected_dtype in schema['dtypes'].items():
                    if col in df.columns:
                        actual_dtype = str(df[col].dtype)
                        if not actual_dtype.startswith(expected_dtype):
                            errors.append(f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'")

            # Check value ranges
            if 'ranges' in schema:
                for col, range_spec in schema['ranges'].items():
                    if col in df.columns:
                        col_min = df[col].min()
                        col_max = df[col].max()

                        if 'min' in range_spec and col_min < range_spec['min']:
                            errors.append(f"Column '{col}' has values below minimum {range_spec['min']}")

                        if 'max' in range_spec and col_max > range_spec['max']:
                            errors.append(f"Column '{col}' has values above maximum {range_spec['max']}")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 ** 2,
                'dtypes': df.dtypes.value_counts().to_dict(),
            }
        }

    def profile_dataframe(self, df: Any) -> Dict[str, Any]:
        """Generate comprehensive DataFrame profile"""
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        profile = {
            'basic_info': {
                'shape': df.shape,
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 ** 2,
                'columns': list(df.columns),
            },
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': {
                'total': int(df.duplicated().sum()),
                'percentage': float(df.duplicated().sum() / len(df) * 100),
            },
        }

        # Numeric columns analysis
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            profile['numeric_summary'] = numeric_df.describe().to_dict()
            profile['correlations'] = numeric_df.corr().to_dict()

        # Categorical columns analysis
        categorical_df = df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            profile['categorical_summary'] = {
                col: {
                    'unique': int(df[col].nunique()),
                    'top_values': df[col].value_counts().head(10).to_dict(),
                }
                for col in categorical_df.columns
            }

        return profile

    def merge_dataframes(self, dfs: List[Any], **kwargs) -> Any:
        """Merge multiple DataFrames"""
        import pandas as pd

        if not dfs:
            return pd.DataFrame()

        # Determine merge strategy
        how = kwargs.get('how', 'inner')
        on = kwargs.get('on', None)

        if len(dfs) == 1:
            return dfs[0]

        # Sequential merge
        result = dfs[0]
        for df in dfs[1:]:
            if on:
                result = pd.merge(result, df, how=how, on=on)
            else:
                # Try to find common columns
                common_cols = set(result.columns) & set(df.columns)
                if common_cols:
                    result = pd.merge(result, df, how=how, on=list(common_cols))
                else:
                    # Concat if no common columns
                    result = pd.concat([result, df], axis=1)

        return result

    def to_parquet_partitioned(self, df: Any, path: str, partition_cols: List[str]) -> None:
        """Save DataFrame as partitioned Parquet dataset"""
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        df.to_parquet(
            path,
            engine='pyarrow',
            partition_cols=partition_cols,
            compression='snappy',
        )

        logger.info(f"Saved partitioned dataset to {path}")

    def read_large_csv(self, filepath: str, chunksize: int = 10000, **kwargs) -> Any:
        """Read large CSV file in chunks"""
        import pandas as pd

        # Process in chunks
        chunks = []

        for chunk in pd.read_csv(filepath, chunksize=chunksize, **kwargs):
            # Process chunk if needed
            if 'process_chunk' in kwargs:
                chunk = kwargs['process_chunk'](chunk)

            chunks.append(chunk)

        # Combine all chunks
        return pd.concat(chunks, ignore_index=True)