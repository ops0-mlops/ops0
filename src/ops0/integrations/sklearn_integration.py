"""
ops0 Scikit-Learn Integration

Seamless integration with scikit-learn models and pipelines.
Automatic serialization, model registry, and deployment support.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import warnings

from .base import (
    BaseIntegration,
    ModelWrapper,
    DatasetWrapper,
    IntegrationCapability,
    SerializationHandler,
    ModelMetadata,
    DatasetMetadata,
)

logger = logging.getLogger(__name__)


class ScikitLearnSerializer(SerializationHandler):
    """Scikit-learn specific serialization using joblib"""

    def serialize(self, obj: Any) -> bytes:
        """Serialize sklearn model using joblib"""
        import io
        try:
            import joblib
        except ImportError:
            # Fallback to pickle
            import pickle
            buffer = io.BytesIO()
            pickle.dump(obj, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            return buffer.getvalue()

        buffer = io.BytesIO()
        joblib.dump(obj, buffer, compress=3)
        return buffer.getvalue()

    def deserialize(self, data: bytes) -> Any:
        """Deserialize sklearn model"""
        import io
        try:
            import joblib
            buffer = io.BytesIO(data)
            return joblib.load(buffer)
        except ImportError:
            # Fallback to pickle
            import pickle
            buffer = io.BytesIO(data)
            return pickle.load(buffer)

    def get_format(self) -> str:
        return "joblib"


class ScikitLearnModelWrapper(ModelWrapper):
    """Wrapper for scikit-learn models"""

    def _extract_metadata(self) -> ModelMetadata:
        """Extract metadata from sklearn model"""
        import sklearn

        model_type = type(self.model).__name__
        params = {}

        # Get model parameters if available
        if hasattr(self.model, 'get_params'):
            params = self.model.get_params()

        # Get feature information if available
        input_shape = None
        output_shape = None

        if hasattr(self.model, 'n_features_in_'):
            input_shape = (None, self.model.n_features_in_)

        if hasattr(self.model, 'n_outputs_'):
            output_shape = (None, self.model.n_outputs_)
        elif hasattr(self.model, 'classes_'):
            output_shape = (None, len(self.model.classes_))

        return ModelMetadata(
            framework='sklearn',
            framework_version=sklearn.__version__,
            model_type=model_type,
            input_shape=input_shape,
            output_shape=output_shape,
            parameters=params,
        )

    def predict(self, data: Any, **kwargs) -> Any:
        """Run inference using sklearn model"""
        # Handle different prediction methods
        if hasattr(self.model, 'predict_proba') and kwargs.get('probabilities', False):
            return self.model.predict_proba(data)
        elif hasattr(self.model, 'decision_function') and kwargs.get('decision_function', False):
            return self.model.decision_function(data)
        elif hasattr(self.model, 'transform') and kwargs.get('transform', False):
            return self.model.transform(data)
        else:
            return self.model.predict(data)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}


class ScikitLearnDatasetWrapper(DatasetWrapper):
    """Wrapper for datasets compatible with sklearn"""

    def _extract_metadata(self) -> DatasetMetadata:
        """Extract metadata from dataset"""
        import numpy as np

        # Handle different data types
        if hasattr(self.data, 'shape'):
            shape = self.data.shape
            n_samples = shape[0] if len(shape) > 0 else 1
            n_features = shape[1] if len(shape) > 1 else 1
        else:
            # Convert to numpy to get shape
            arr = np.asarray(self.data)
            shape = arr.shape
            n_samples = shape[0] if len(shape) > 0 else 1
            n_features = shape[1] if len(shape) > 1 else 1

        # Detect dtypes
        dtypes = {}
        if hasattr(self.data, 'dtype'):
            dtypes['data'] = str(self.data.dtype)

        return DatasetMetadata(
            format='numpy',
            shape=shape,
            dtypes=dtypes,
            n_samples=n_samples,
            n_features=n_features,
        )

    def to_numpy(self) -> Any:
        """Convert to numpy array"""
        import numpy as np
        return np.asarray(self.data)

    def split(self, train_size: float = 0.8) -> Tuple[Any, Any]:
        """Split into train/test sets"""
        try:
            from sklearn.model_selection import train_test_split

            # Generate indices
            n_samples = self.data.shape[0] if hasattr(self.data, 'shape') else len(self.data)
            indices = list(range(n_samples))

            train_idx, test_idx = train_test_split(
                indices,
                train_size=train_size,
                random_state=42
            )

            # Split data
            if hasattr(self.data, '__getitem__'):
                train_data = self.data[train_idx]
                test_data = self.data[test_idx]
            else:
                # Convert to numpy first
                import numpy as np
                arr = np.asarray(self.data)
                train_data = arr[train_idx]
                test_data = arr[test_idx]

            return train_data, test_data

        except ImportError:
            # Simple split without sklearn
            import numpy as np
            arr = np.asarray(self.data)
            split_idx = int(len(arr) * train_size)
            return arr[:split_idx], arr[split_idx:]

    def sample(self, n: int, random_state: int = None) -> Any:
        """Sample n rows from dataset"""
        import numpy as np

        arr = np.asarray(self.data)
        if random_state is not None:
            np.random.seed(random_state)

        indices = np.random.choice(len(arr), size=min(n, len(arr)), replace=False)
        return arr[indices]


class ScikitLearnIntegration(BaseIntegration):
    """Integration for scikit-learn ML framework"""

    @property
    def is_available(self) -> bool:
        """Check if sklearn is installed"""
        try:
            import sklearn
            return True
        except ImportError:
            return False

    def _define_capabilities(self) -> List[IntegrationCapability]:
        """Define sklearn capabilities"""
        return [
            IntegrationCapability.TRAINING,
            IntegrationCapability.INFERENCE,
            IntegrationCapability.SERIALIZATION,
            IntegrationCapability.MODEL_REGISTRY,
            IntegrationCapability.DATA_VALIDATION,
        ]

    def get_version(self) -> str:
        """Get sklearn version"""
        try:
            import sklearn
            return sklearn.__version__
        except ImportError:
            return "not installed"

    def wrap_model(self, model: Any) -> ModelWrapper:
        """Wrap sklearn model"""
        return ScikitLearnModelWrapper(model)

    def wrap_dataset(self, data: Any) -> DatasetWrapper:
        """Wrap dataset for sklearn"""
        return ScikitLearnDatasetWrapper(data)

    def get_serialization_handler(self) -> SerializationHandler:
        """Get sklearn serialization handler"""
        if self._serialization_handler is None:
            self._serialization_handler = ScikitLearnSerializer()
        return self._serialization_handler

    def validate_model(self, model: Any) -> Dict[str, bool]:
        """Validate sklearn model"""
        checks = {
            'has_fit': hasattr(model, 'fit'),
            'has_predict': hasattr(model, 'predict'),
            'is_fitted': False,
        }

        # Check if model is fitted
        try:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(model)
            checks['is_fitted'] = True
        except:
            pass

        return checks

    def create_pipeline(self, steps: List[Tuple[str, Any]]) -> Any:
        """Create sklearn pipeline"""
        from sklearn.pipeline import Pipeline
        return Pipeline(steps)

    def cross_validate(self, model: Any, X: Any, y: Any, cv: int = 5, **kwargs) -> Dict[str, Any]:
        """Run cross-validation"""
        from sklearn.model_selection import cross_validate as sklearn_cv

        # Default scoring metrics
        scoring = kwargs.get('scoring', ['accuracy', 'f1_macro', 'roc_auc_ovr'])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = sklearn_cv(
                model, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
                **kwargs
            )

        # Convert to regular dict and compute means
        return {
            'metrics': {key: float(val.mean()) for key, val in results.items()},
            'std': {key: float(val.std()) for key, val in results.items()},
            'raw_scores': {key: val.tolist() for key, val in results.items()},
        }

    def auto_select_model(self, task: str = 'classification', **kwargs) -> Any:
        """Auto-select appropriate model for task"""
        if task == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
        elif task == 'regression':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=100, random_state=42, **kwargs)
        elif task == 'clustering':
            from sklearn.cluster import KMeans
            return KMeans(n_clusters=kwargs.get('n_clusters', 8), random_state=42)
        else:
            raise ValueError(f"Unknown task: {task}")

    def explain_predictions(self, model: Any, data: Any, **kwargs) -> Dict[str, Any]:
        """Generate model explanations using SHAP or similar"""
        try:
            import shap

            # Create explainer
            explainer = shap.Explainer(model, data)
            shap_values = explainer(data)

            return {
                'shap_values': shap_values.values,
                'base_values': shap_values.base_values,
                'feature_names': shap_values.feature_names,
            }

        except ImportError:
            logger.warning("SHAP not available for explanations")

            # Fallback to feature importances if available
            if hasattr(model, 'feature_importances_'):
                return {
                    'feature_importances': model.feature_importances_.tolist()
                }

            return {'error': 'No explanation method available'}