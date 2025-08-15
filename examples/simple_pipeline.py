"""
Simple example pipeline demonstrating ops0's zero-configuration approach.
This pipeline loads data, processes it, and trains a model.
"""
import ops0
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


@ops0.step
def load_iris_data():
    """Load the classic Iris dataset"""
    from sklearn.datasets import load_iris

    iris = load_iris()
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['target'] = iris.target

    print(f"✅ Loaded {len(df)} samples")
    return df


@ops0.step
def prepare_features(df: pd.DataFrame):
    """Prepare features for training"""
    # ops0 automatically handles the dataframe serialization between steps

    # Add some feature engineering
    df['sepal_ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']
    df['petal_ratio'] = df['petal length (cm)'] / df['petal width (cm)']

    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"✅ Prepared {len(X_train)} training samples, {len(X_test)} test samples")

    # ops0 handles returning multiple values
    return X_train, X_test, y_train, y_test


@ops0.step
def train_classifier(X_train, X_test, y_train, y_test):
    """Train a Random Forest classifier"""
    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Model accuracy: {accuracy:.2%}")

    # Save model using ops0's model management
    ops0.save_model(clf, "iris_classifier", {"accuracy": accuracy})

    # Save detailed results
    report = classification_report(y_test, y_pred, output_dict=True)
    ops0.save("classification_report", report)

    return {"accuracy": accuracy, "model_saved": True}


@ops0.pipeline
def iris_classification_pipeline():
    """
    Complete pipeline for Iris classification.

    This demonstrates:
    - Automatic dependency detection (ops0 knows the execution order)
    - Zero configuration (no YAML, no DAG definition)
    - Built-in storage and model management
    """
    # Load data
    data = load_iris_data()

    # Prepare features
    X_train, X_test, y_train, y_test = prepare_features(data)

    # Train and evaluate
    results = train_classifier(X_train, X_test, y_train, y_test)

    return results


if __name__ == "__main__":
    # Run the pipeline
    print("🚀 Starting Iris classification pipeline...\n")

    results = iris_classification_pipeline()

    print(f"\n✅ Pipeline completed!")
    print(f"📊 Results: {results}")

    # Show how to load the saved model
    print("\n🔄 Loading saved model...")
    model = ops0.load_model("iris_classifier")
    print(f"✅ Model loaded: {type(model).__name__}")