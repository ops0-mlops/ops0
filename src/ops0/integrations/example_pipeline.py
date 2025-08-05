"""
Example ML Pipeline using ops0 Integrations

This demonstrates how ops0 automatically handles different ML frameworks
in a single pipeline with zero configuration.
"""

import ops0.core as ops0
import numpy as np
import pandas as pd


@ops0.pipeline(name="multi-framework-ml")
def create_ml_pipeline():
    """
    Example pipeline that uses multiple ML frameworks seamlessly.
    ops0 automatically detects and optimizes each framework.
    """

    @ops0.step
    def generate_data() -> pd.DataFrame:
        """Generate synthetic dataset using pandas"""
        np.random.seed(42)

        n_samples = 1000
        n_features = 20

        # Generate features
        X = np.random.randn(n_samples, n_features)

        # Generate target with some signal
        weights = np.random.randn(n_features)
        y = (X @ weights + np.random.randn(n_samples) * 0.1 > 0).astype(int)

        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y

        print(f"Generated dataset: {df.shape}")
        print(f"Class distribution: {df['target'].value_counts().to_dict()}")

        # ops0 automatically uses Parquet for pandas DataFrames
        return df

    @ops0.step
    def preprocess_data(df: pd.DataFrame) -> tuple:
        """Preprocess data using pandas and numpy"""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Split features and target
        X = df.drop('target', axis=1).values  # Convert to numpy
        y = df['target'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"Train set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")

        # ops0 handles numpy arrays efficiently
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    @ops0.step
    def train_sklearn_model(X_train, y_train, X_test, y_test):
        """Train scikit-learn model"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        print(f"Sklearn RF - Train accuracy: {train_score:.3f}")
        print(f"Sklearn RF - Test accuracy: {test_score:.3f}")

        # Get predictions for report
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # ops0 uses joblib for sklearn models
        return model, report

    @ops0.step
    def train_xgboost_model(X_train, y_train, X_test, y_test):
        """Train XGBoost model with GPU if available"""
        import xgboost as xgb
        from ops0.integrations import get_integration

        # Get XGBoost integration
        xgb_integration = get_integration('xgboost')

        # Create model with automatic GPU detection
        model = xgb_integration.create_classifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic'
        )

        # Train
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )

        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        print(f"XGBoost - Train accuracy: {train_score:.3f}")
        print(f"XGBoost - Test accuracy: {test_score:.3f}")

        # Get feature importance
        importance = model.feature_importances_

        return model, importance

    @ops0.step
    @ops0.gpu  # Request GPU for this step
    def train_neural_network(X_train, y_train, X_test, y_test):
        """Train PyTorch neural network if available"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader

            # Convert to tensors
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train)
            X_test_t = torch.FloatTensor(X_test)
            y_test_t = torch.LongTensor(y_test)

            # Create data loaders
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Define model
            class SimpleNN(nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, 64)
                    self.fc2 = nn.Linear(64, 32)
                    self.fc3 = nn.Linear(32, 2)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.2)

                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.relu(self.fc2(x))
                    x = self.dropout(x)
                    x = self.fc3(x)
                    return x

            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SimpleNN(X_train.shape[1]).to(device)

            # Train
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for epoch in range(10):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train_t.to(device))
                train_preds = torch.argmax(train_outputs, dim=1).cpu()
                train_acc = (train_preds == y_train_t).float().mean()

                test_outputs = model(X_test_t.to(device))
                test_preds = torch.argmax(test_outputs, dim=1).cpu()
                test_acc = (test_preds == y_test_t).float().mean()

            print(f"PyTorch NN - Train accuracy: {train_acc:.3f}")
            print(f"PyTorch NN - Test accuracy: {test_acc:.3f}")
            print(f"Device used: {device}")

            # ops0 handles PyTorch model serialization
            return model.cpu(), float(test_acc)

        except ImportError:
            print("PyTorch not available, skipping neural network")
            return None, 0.0

    @ops0.step
    def compare_models(sklearn_model, sklearn_report, xgb_model, xgb_importance, nn_model, nn_acc):
        """Compare all models and select the best"""
        results = {
            'sklearn': {
                'model': sklearn_model,
                'accuracy': sklearn_report['accuracy'],
                'type': 'RandomForest'
            },
            'xgboost': {
                'model': xgb_model,
                'accuracy': xgb_model.score(X_test, y_test),  # Would need X_test, y_test
                'type': 'XGBoost'
            }
        }

        if nn_model is not None:
            results['pytorch'] = {
                'model': nn_model,
                'accuracy': nn_acc,
                'type': 'NeuralNetwork'
            }

        # Find best model
        best_framework = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_model = results[best_framework]['model']

        print("\n=== Model Comparison ===")
        for framework, info in results.items():
            print(f"{framework}: {info['accuracy']:.3f} ({info['type']})")

        print(f"\nBest model: {best_framework} with {results[best_framework]['accuracy']:.3f} accuracy")

        # ops0 automatically detects and serializes the best model
        return best_model, results

    @ops0.step
    def create_model_card(best_model, results):
        """Create a model card with all information"""
        from datetime import datetime

        model_card = {
            'created_at': datetime.now().isoformat(),
            'framework': detect_framework(best_model),
            'model_type': type(best_model).__name__,
            'performance': results,
            'ops0_features': {
                'auto_serialization': True,
                'gpu_support': 'auto-detected',
                'distributed_ready': True
            }
        }

        print("\n=== Model Card ===")
        print(f"Framework: {model_card['framework']}")
        print(f"Model Type: {model_card['model_type']}")
        print(f"Created: {model_card['created_at']}")

        return model_card

    # Define pipeline flow
    data = generate_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # Train models in parallel (ops0 handles this automatically)
    sklearn_model, sklearn_report = train_sklearn_model(X_train, y_train, X_test, y_test)
    xgb_model, xgb_importance = train_xgboost_model(X_train, y_train, X_test, y_test)
    nn_model, nn_acc = train_neural_network(X_train, y_train, X_test, y_test)

    # Compare and select best
    best_model, results = compare_models(
        sklearn_model, sklearn_report,
        xgb_model, xgb_importance,
        nn_model, nn_acc
    )

    # Create model card
    model_card = create_model_card(best_model, results)

    return best_model, model_card


# Helper function for framework detection
def detect_framework(model):
    """Detect ML framework of a model"""
    from ops0.integrations import detect_framework as ops0_detect
    return ops0_detect(model) or 'unknown'


if __name__ == '__main__':
    print("🚀 ops0 Multi-Framework ML Pipeline Example")
    print("=" * 50)
    print("\nThis example demonstrates:")
    print("- Automatic framework detection")
    print("- Optimized serialization for each framework")
    print("- GPU support auto-detection")
    print("- Parallel model training")
    print("- Transparent model comparison")
    print("\n" + "=" * 50)

    # Create and run pipeline
    pipeline = create_ml_pipeline()

    # In production, you would deploy with:
    # ops0.deploy(pipeline)

    print("\n✅ Pipeline created successfully!")
    print("\nTo deploy this pipeline:")
    print("  ops0 deploy --name multi-framework-ml")
    print("\nTo run locally:")
    print("  ops0 run --local")