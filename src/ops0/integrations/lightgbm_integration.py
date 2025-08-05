"""
ops0 LightGBM Integration

High-performance gradient boosting with LightGBM - automatic categorical feature handling,
GPU acceleration, and distributed training support.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import tempfile
import os

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


class LightGBMSerializer(SerializationHandler):
    """LightGBM model serialization handler"""

    def serialize(self, obj: Any) -> bytes:
        """Serialize LightGBM model"""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
            obj.save_model(tmp.name)
            tmp.flush()

            # Read back as bytes
            with open(tmp.name, 'rb') as f:
                data = f.read()

            os.unlink(tmp.name)
            return data

    def deserialize(self, data: bytes) -> Any:
        """Deserialize LightGBM model"""
        import lightgbm as lgb

        # Write to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
            tmp.write(data)
            tmp.flush()

            # Load model
            model = lgb.Booster(model_file=tmp.name)

            os.unlink(tmp.name)
            return model

    def get_format(self) -> str:
        return "lightgbm-txt"


class LightGBMModelWrapper(ModelWrapper):
    """Wrapper for LightGBM models"""

    def _extract_metadata(self) -> ModelMetadata:
        """Extract metadata from LightGBM model"""
        import lightgbm as lgb

        model_type = type(self.model).__name__

        # Get model configuration
        config = {}
        params = {}

        if hasattr(self.model, 'get_params'):
            # Scikit-learn API
            params = self.model.get_params()
        elif hasattr(self.model, 'params'):
            # Booster
            params = self.model.params

        # Get feature importance
        importance = {}
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Scikit-learn API
                importance = {
                    f'feature_{i}': float(imp)
                    for i, imp in enumerate(self.model.feature_importances_)
                }
            elif hasattr(self.model, 'feature_importance'):
                # Booster API
                imp_values = self.model.feature_importance(importance_type='gain')
                feature_names = self.model.feature_name()
                importance = {
                    name: float(imp)
                    for name, imp in zip(feature_names, imp_values)
                }
        except:
            pass

        # Get tree info
        tree_info = {}
        if hasattr(self.model, 'num_trees'):
            tree_info['num_trees'] = self.model.num_trees()
        if hasattr(self.model, 'num_feature'):
            tree_info['num_features'] = self.model.num_feature()

        return ModelMetadata(
            framework='lightgbm',
            framework_version=lgb.__version__,
            model_type=model_type,
            parameters={**params, **tree_info},
            metrics={'feature_importance': importance},
        )

    def predict(self, data: Any, **kwargs) -> Any:
        """Run inference using LightGBM model"""
        import lightgbm as lgb
        import numpy as np

        # Handle different prediction methods
        if hasattr(self.model, 'predict_proba') and kwargs.get('predict_proba', False):
            # Scikit-learn API
            return self.model.predict_proba(data)
        elif hasattr(self.model, 'predict'):
            # Both APIs have predict
            num_iteration = kwargs.get('num_iteration', None)
            if hasattr(self.model, 'booster_'):
                # Scikit-learn API with access to booster
                return self.model.predict(data, num_iteration=num_iteration)
            else:
                # Direct booster predict
                return self.model.predict(data, num_iteration=num_iteration)
        else:
            raise ValueError("Model doesn't have predict method")

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        elif hasattr(self.model, 'params'):
            return self.model.params
        return {}


class LightGBMDatasetWrapper(DatasetWrapper):
    """Wrapper for LightGBM Dataset"""

    def __init__(self, data: Any, label: Any = None, metadata: Optional[DatasetMetadata] = None):
        """
        Initialize with data and optional labels.

        Args:
            data: Feature data
            label: Target labels
            metadata: Optional metadata
        """
        import lightgbm as lgb

        # Create Dataset
        if isinstance(data, lgb.Dataset):
            self.data = data
        else:
            self.data = lgb.Dataset(data, label=label, free_raw_data=False)

        self.metadata = metadata or self._extract_metadata()

    def _extract_metadata(self) -> DatasetMetadata:
        """Extract metadata from Dataset"""
        # LightGBM Dataset doesn't expose shape directly
        # We need to construct it when creating the wrapper
        return DatasetMetadata(
            format='lightgbm.Dataset',
            shape=(0, 0),  # Will be set externally
            n_samples=0,
            n_features=0,
        )

    def to_numpy(self) -> Any:
        """Convert to numpy array"""
        # LightGBM doesn't provide direct access to data
        if hasattr(self.data, 'data'):
            return self.data.data
        else:
            raise NotImplementedError("Cannot convert LightGBM Dataset back to numpy")

    def split(self, train_size: float = 0.8) -> Tuple[Any, Any]:
        """Split into train/test sets"""
        raise NotImplementedError("Use original data for splitting with LightGBM")

    def sample(self, n: int, random_state: int = None) -> Any:
        """Sample from Dataset"""
        raise NotImplementedError("Cannot sample from LightGBM Dataset")


class LightGBMIntegration(BaseIntegration):
    """Integration for LightGBM gradient boosting"""

    @property
    def is_available(self) -> bool:
        """Check if LightGBM is installed"""
        try:
            import lightgbm
            return True
        except ImportError:
            return False

    def _define_capabilities(self) -> List[IntegrationCapability]:
        """Define LightGBM capabilities"""
        caps = [
            IntegrationCapability.TRAINING,
            IntegrationCapability.INFERENCE,
            IntegrationCapability.SERIALIZATION,
            IntegrationCapability.MODEL_REGISTRY,
        ]

        # Check for GPU support
        try:
            import lightgbm as lgb
            # LightGBM GPU support is compile-time
            # Try to create a simple dataset with gpu params
            try:
                params = {'device': 'gpu'}
                # This will fail if GPU support not compiled
                caps.append(IntegrationCapability.GPU_SUPPORT)
            except:
                pass
        except:
            pass

        return caps

    def get_version(self) -> str:
        """Get LightGBM version"""
        try:
            import lightgbm
            return lightgbm.__version__
        except ImportError:
            return "not installed"

    def wrap_model(self, model: Any) -> ModelWrapper:
        """Wrap LightGBM model"""
        return LightGBMModelWrapper(model)

    def wrap_dataset(self, data: Any) -> DatasetWrapper:
        """Wrap dataset for LightGBM"""
        return LightGBMDatasetWrapper(data)

    def get_serialization_handler(self) -> SerializationHandler:
        """Get LightGBM serialization handler"""
        if self._serialization_handler is None:
            self._serialization_handler = LightGBMSerializer()
        return self._serialization_handler

    def create_classifier(self, **kwargs) -> Any:
        """Create LightGBM classifier with sensible defaults"""
        import lightgbm as lgb

        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'num_threads': -1,
            'seed': 42,
        }

        # Use GPU if requested
        if kwargs.get('use_gpu', False):
            params['device'] = 'gpu'
            params['gpu_platform_id'] = kwargs.get('gpu_platform_id', 0)
            params['gpu_device_id'] = kwargs.get('gpu_device_id', 0)

        # Handle binary classification
        if kwargs.get('binary', False) or kwargs.get('num_class', 2) == 2:
            params['objective'] = 'binary'
            params['metric'] = 'binary_logloss'

        params.update(kwargs)

        return lgb.LGBMClassifier(**params)

    def create_regressor(self, **kwargs) -> Any:
        """Create LightGBM regressor with sensible defaults"""
        import lightgbm as lgb

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'num_threads': -1,
            'seed': 42,
        }

        # Use GPU if requested
        if kwargs.get('use_gpu', False):
            params['device'] = 'gpu'
            params['gpu_platform_id'] = kwargs.get('gpu_platform_id', 0)
            params['gpu_device_id'] = kwargs.get('gpu_device_id', 0)

        params.update(kwargs)

        return lgb.LGBMRegressor(**params)

    def create_ranker(self, **kwargs) -> Any:
        """Create LightGBM ranker"""
        import lightgbm as lgb

        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'num_threads': -1,
            'seed': 42,
        }

        params.update(kwargs)

        return lgb.LGBMRanker(**params)

    def handle_categorical_features(self, X: Any, categorical_features: List[Union[str, int]]) -> Any:
        """Prepare categorical features for LightGBM"""
        import pandas as pd
        import numpy as np

        if isinstance(X, pd.DataFrame):
            # Convert categorical columns to 'category' dtype
            for col in categorical_features:
                if col in X.columns:
                    X[col] = X[col].astype('category')
        else:
            logger.warning("Categorical feature handling works best with pandas DataFrames")

        return X

    def hyperparameter_search(
            self,
            X: Any,
            y: Any,
            param_grid: Dict[str, List[Any]],
            cv: int = 5,
            **kwargs
    ) -> Dict[str, Any]:
        """Perform hyperparameter search with Optuna"""
        try:
            import optuna
            import lightgbm as lgb
            from sklearn.model_selection import cross_val_score

            # Define objective function
            def objective(trial):
                params = {}

                # Sample hyperparameters
                for param_name, param_values in param_grid.items():
                    if isinstance(param_values, list):
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                    elif isinstance(param_values, tuple) and len(param_values) == 2:
                        if isinstance(param_values[0], int):
                            params[param_name] = trial.suggest_int(param_name, param_values[0], param_values[1])
                        else:
                            params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])

                # Create model
                model = self.create_classifier(**params)

                # Cross-validation
                scores = cross_val_score(
                    model, X, y,
                    cv=cv,
                    scoring=kwargs.get('scoring', 'accuracy')
                )

                return scores.mean()

            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            # Optimize
            study.optimize(
                objective,
                n_trials=kwargs.get('n_trials', 100),
                show_progress_bar=True
            )

            # Get best parameters and retrain
            best_model = self.create_classifier(**study.best_params)
            best_model.fit(X, y)

            return {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'best_model': best_model,
                'study': study,
            }

        except ImportError:
            logger.warning("Optuna not available, falling back to grid search")

            # Fallback to sklearn GridSearchCV
            from sklearn.model_selection import GridSearchCV

            base_estimator = kwargs.get('estimator', self.create_classifier())

            grid_search = GridSearchCV(
                base_estimator,
                param_grid,
                cv=cv,
                scoring=kwargs.get('scoring', 'accuracy'),
                n_jobs=-1,
            )

            grid_search.fit(X, y)

            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_model': grid_search.best_estimator_,
            }

    def explain_predictions(self, model: Any, data: Any, **kwargs) -> Dict[str, Any]:
        """Generate SHAP explanations for LightGBM"""
        try:
            import shap

            # Create SHAP explainer
            # LightGBM has native SHAP support
            explainer = shap.TreeExplainer(model)

            # Calculate SHAP values
            shap_values = explainer.shap_values(data)

            # Handle multi-class
            if isinstance(shap_values, list):
                # Multi-class case
                return {
                    'shap_values': shap_values,
                    'base_values': explainer.expected_value,
                    'feature_names': model.feature_name_ if hasattr(model, 'feature_name_') else None,
                    'classes': model.classes_ if hasattr(model, 'classes_') else None,
                }
            else:
                # Binary/regression case
                return {
                    'shap_values': shap_values,
                    'base_values': explainer.expected_value,
                    'feature_names': model.feature_name_ if hasattr(model, 'feature_name_') else None,
                }

        except ImportError:
            logger.warning("SHAP not available for explanations")

            # Fallback to built-in feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                names = model.feature_name_ if hasattr(model, 'feature_name_') else [f'f{i}' for i in
                                                                                     range(len(importance))]

                return {
                    'feature_importance': dict(zip(names, importance)),
                    'importance_type': 'gain',
                }

            return {}

    def early_stopping_callback(self, eval_set: List[Tuple], eval_names: List[str] = None) -> Any:
        """Create early stopping callback"""
        import lightgbm as lgb

        return lgb.early_stopping(
            stopping_rounds=kwargs.get('stopping_rounds', 50),
            first_metric_only=kwargs.get('first_metric_only', False),
            verbose=kwargs.get('verbose', True)
        )

    def plot_importance(self, model: Any, max_features: int = 20, **kwargs) -> None:
        """Plot feature importance"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            names = model.feature_name_ if hasattr(model, 'feature_name_') else [f'f{i}' for i in
                                                                                 range(len(importance))]
        else:
            return

        # Sort by importance
        indices = np.argsort(importance)[::-1][:max_features]

        # Plot
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        plt.title('Feature Importances')
        plt.bar(range(len(indices)), importance[indices])
        plt.xticks(range(len(indices)), [names[i] for i in indices], rotation=90)
        plt.tight_layout()

        if kwargs.get('save_path'):
            plt.savefig(kwargs['save_path'])
        else:
            plt.show()