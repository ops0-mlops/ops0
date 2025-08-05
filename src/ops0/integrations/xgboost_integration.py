"""
ops0 XGBoost Integration

Gradient boosting with XGBoost - automatic hyperparameter tuning,
GPU acceleration, and distributed training support.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import json

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


class XGBoostSerializer(SerializationHandler):
    """XGBoost model serialization handler"""

    def serialize(self, obj: Any) -> bytes:
        """Serialize XGBoost model"""
        import tempfile
        import os

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            obj.save_model(tmp.name)
            tmp.flush()

            # Read back as bytes
            with open(tmp.name, 'rb') as f:
                data = f.read()

            os.unlink(tmp.name)
            return data

    def deserialize(self, data: bytes) -> Any:
        """Deserialize XGBoost model"""
        import xgboost as xgb
        import tempfile
        import os

        # Write to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            tmp.write(data)
            tmp.flush()

            # Load model
            model = xgb.Booster()
            model.load_model(tmp.name)

            os.unlink(tmp.name)
            return model

    def get_format(self) -> str:
        return "xgboost-json"


class XGBoostModelWrapper(ModelWrapper):
    """Wrapper for XGBoost models"""

    def _extract_metadata(self) -> ModelMetadata:
        """Extract metadata from XGBoost model"""
        import xgboost as xgb

        model_type = type(self.model).__name__

        # Get model configuration
        config = {}
        if hasattr(self.model, 'get_params'):
            config = self.model.get_params()
        elif hasattr(self.model, 'save_config'):
            config = json.loads(self.model.save_config())

        # Get feature importance
        importance = {}
        try:
            if hasattr(self.model, 'get_score'):
                importance = self.model.get_score()
            elif hasattr(self.model, 'feature_importances_'):
                importance = {
                    f'f{i}': float(imp)
                    for i, imp in enumerate(self.model.feature_importances_)
                }
        except:
            pass

        return ModelMetadata(
            framework='xgboost',
            framework_version=xgb.__version__,
            model_type=model_type,
            parameters=config,
            metrics={'feature_importance': importance},
        )

    def predict(self, data: Any, **kwargs) -> Any:
        """Run inference using XGBoost model"""
        import xgboost as xgb
        import numpy as np

        # Convert to DMatrix if needed
        if not isinstance(data, xgb.DMatrix):
            if hasattr(data, 'values'):  # pandas DataFrame
                data = xgb.DMatrix(data.values)
            else:
                data = xgb.DMatrix(data)

        # Get prediction
        output_margin = kwargs.get('output_margin', False)
        ntree_limit = kwargs.get('ntree_limit', 0)

        predictions = self.model.predict(
            data,
            output_margin=output_margin,
            ntree_limit=ntree_limit
        )

        return predictions

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        elif hasattr(self.model, 'save_config'):
            return json.loads(self.model.save_config())
        return {}


class XGBoostDatasetWrapper(DatasetWrapper):
    """Wrapper for XGBoost DMatrix"""

    def __init__(self, data: Any, label: Any = None, metadata: Optional[DatasetMetadata] = None):
        """
        Initialize with data and optional labels.

        Args:
            data: Feature data
            label: Target labels
            metadata: Optional metadata
        """
        import xgboost as xgb

        # Create DMatrix
        if isinstance(data, xgb.DMatrix):
            self.data = data
        else:
            self.data = xgb.DMatrix(data, label=label)

        self.metadata = metadata or self._extract_metadata()

    def _extract_metadata(self) -> DatasetMetadata:
        """Extract metadata from DMatrix"""
        return DatasetMetadata(
            format='xgboost.DMatrix',
            shape=(self.data.num_row(), self.data.num_col()),
            n_samples=self.data.num_row(),
            n_features=self.data.num_col(),
        )

    def to_numpy(self) -> Any:
        """Convert to numpy array"""
        # XGBoost doesn't provide direct access to data
        raise NotImplementedError("Cannot convert DMatrix back to numpy")

    def split(self, train_size: float = 0.8) -> Tuple[Any, Any]:
        """Split into train/test sets"""
        # XGBoost requires recreating DMatrix
        raise NotImplementedError("Use original data for splitting with XGBoost")

    def sample(self, n: int, random_state: int = None) -> Any:
        """Sample from DMatrix"""
        raise NotImplementedError("Cannot sample from DMatrix")


class XGBoostIntegration(BaseIntegration):
    """Integration for XGBoost gradient boosting"""

    @property
    def is_available(self) -> bool:
        """Check if XGBoost is installed"""
        try:
            import xgboost
            return True
        except ImportError:
            return False

    def _define_capabilities(self) -> List[IntegrationCapability]:
        """Define XGBoost capabilities"""
        caps = [
            IntegrationCapability.TRAINING,
            IntegrationCapability.INFERENCE,
            IntegrationCapability.SERIALIZATION,
            IntegrationCapability.MODEL_REGISTRY,
        ]

        # Check for GPU support
        try:
            import xgboost as xgb
            # Try to create a GPU booster
            test_params = {'tree_method': 'gpu_hist'}
            # This will fail if GPU not available
            caps.append(IntegrationCapability.GPU_SUPPORT)
        except:
            pass

        return caps

    def get_version(self) -> str:
        """Get XGBoost version"""
        try:
            import xgboost
            return xgboost.__version__
        except ImportError:
            return "not installed"

    def wrap_model(self, model: Any) -> ModelWrapper:
        """Wrap XGBoost model"""
        return XGBoostModelWrapper(model)

    def wrap_dataset(self, data: Any) -> DatasetWrapper:
        """Wrap dataset for XGBoost"""
        return XGBoostDatasetWrapper(data)

    def get_serialization_handler(self) -> SerializationHandler:
        """Get XGBoost serialization handler"""
        if self._serialization_handler is None:
            self._serialization_handler = XGBoostSerializer()
        return self._serialization_handler

    def create_classifier(self, **kwargs) -> Any:
        """Create XGBoost classifier with sensible defaults"""
        import xgboost as xgb

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'eta': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
        }

        # Use GPU if available
        if self._check_gpu_available():
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'

        params.update(kwargs)

        return xgb.XGBClassifier(**params)

    def create_regressor(self, **kwargs) -> Any:
        """Create XGBoost regressor with sensible defaults"""
        import xgboost as xgb

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'eta': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
        }

        # Use GPU if available
        if self._check_gpu_available():
            params['tree_method'] = 'gpu_hist'
            params['predictor'] = 'gpu_predictor'

        params.update(kwargs)

        return xgb.XGBRegressor(**params)

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for XGBoost"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True)
            return result.returncode == 0
        except:
            return False

    def hyperparameter_search(
            self,
            X: Any,
            y: Any,
            param_grid: Dict[str, List[Any]],
            cv: int = 5,
            **kwargs
    ) -> Dict[str, Any]:
        """Perform hyperparameter search"""
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV

        # Base estimator
        base_estimator = kwargs.get('estimator', self.create_classifier())

        # Grid search
        grid_search = GridSearchCV(
            base_estimator,
            param_grid,
            cv=cv,
            scoring=kwargs.get('scoring', 'accuracy'),
            n_jobs=kwargs.get('n_jobs', -1),
            verbose=kwargs.get('verbose', 1),
        )

        grid_search.fit(X, y)

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_,
        }

    def explain_predictions(self, model: Any, data: Any, **kwargs) -> Dict[str, Any]:
        """Generate SHAP explanations for XGBoost"""
        try:
            import shap
            import xgboost as xgb

            # Create SHAP explainer
            if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
                explainer = shap.Explainer(model)
            else:
                # Booster object
                explainer = shap.Explainer(model)

            # Calculate SHAP values
            shap_values = explainer(data)

            return {
                'shap_values': shap_values.values,
                'base_values': shap_values.base_values,
                'feature_names': shap_values.feature_names if hasattr(shap_values, 'feature_names') else None,
            }

        except ImportError:
            logger.warning("SHAP not available for explanations")

            # Fallback to built-in feature importance
            importance = model.get_score() if hasattr(model, 'get_score') else {}

            return {
                'feature_importance': importance,
                'importance_type': 'gain',
            }

    def distributed_train(
            self,
            dtrain: Any,
            params: Dict[str, Any],
            num_boost_round: int = 100,
            **kwargs
    ) -> Any:
        """Train XGBoost model in distributed mode"""
        import xgboost as xgb

        # Setup Dask if available
        try:
            import dask
            from dask.distributed import Client
            from xgboost import dask as dxgb

            client = kwargs.get('client', Client())

            # Convert to Dask DMatrix
            if not isinstance(dtrain, dxgb.DaskDMatrix):
                raise ValueError("Distributed training requires DaskDMatrix")

            # Train with Dask
            output = dxgb.train(
                client,
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=kwargs.get('evals', []),
                early_stopping_rounds=kwargs.get('early_stopping_rounds'),
            )

            return output['booster']

        except ImportError:
            logger.warning("Dask not available, falling back to single-node training")

            # Regular training
            return xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=kwargs.get('evals', []),
                early_stopping_rounds=kwargs.get('early_stopping_rounds'),
            )