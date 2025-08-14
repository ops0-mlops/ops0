"""Describe command for ops0 - show detailed information about resources."""

import time
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from rich import box

from ..core import Pipeline
from ..core.config import Config
from ..registry import Registry

from .utils import (
    ensure_project_initialized,
    print_error,
    print_info,
    format_time_ago,
    format_duration,
    format_bytes
)

console = Console()


def describe(
        resource_type: str = typer.Argument(..., help="Resource type (pipeline, step, model, deployment)"),
        name: str = typer.Argument(..., help="Resource name"),
        format: str = typer.Option("rich", "--format", "-f", help="Output format (rich, json, yaml)"),
        show_code: bool = typer.Option(False, "--code", "-c", help="Show source code for steps"),
        show_history: bool = typer.Option(False, "--history", "-h", help="Show resource history"),
        show_metrics: bool = typer.Option(False, "--metrics", "-m", help="Show performance metrics"),
):
    """
    Show detailed information about an ops0 resource.

    Examples:
        ops0 describe pipeline fraud-detector
        ops0 describe step preprocess
        ops0 describe model fraud-detector-rf
        ops0 describe deployment fraud-detector-prod
        ops0 describe pipeline my-pipeline --code
        ops0 describe step train --metrics
    """
    # Ensure we're in an ops0 project
    project_root = ensure_project_initialized()

    # Get registry
    registry = Registry(project_root)

    # Get resource details
    if resource_type in ["pipeline", "pipelines"]:
        resource = describe_pipeline(registry, name, show_code, show_history, show_metrics)
    elif resource_type in ["step", "steps"]:
        resource = describe_step(registry, name, show_code, show_metrics)
    elif resource_type in ["model", "models"]:
        resource = describe_model(registry, name, show_history, show_metrics)
    elif resource_type in ["deployment", "deployments"]:
        resource = describe_deployment(registry, name, show_history, show_metrics)
    else:
        print_error(f"Unknown resource type: {resource_type}")
        console.print("Valid types: pipeline, step, model, deployment")
        raise typer.Exit(1)

    if not resource:
        print_error(f"{resource_type.capitalize()} '{name}' not found")
        raise typer.Exit(1)

    # Display resource
    if format == "json":
        console.print(json.dumps(resource, indent=2))
    elif format == "yaml":
        console.print(yaml.dump(resource, default_flow_style=False))
    else:
        display_resource_rich(resource_type, resource, show_code, show_history, show_metrics)


def describe_pipeline(registry: Registry, name: str, show_code: bool,
                      show_history: bool, show_metrics: bool) -> Optional[Dict[str, Any]]:
    """Get detailed pipeline information."""
    # Mock data for demonstration
    if name != "fraud-detector":
        return None

    pipeline = {
        "name": "fraud-detector",
        "version": "0.2.0",
        "description": "ML pipeline for detecting fraudulent transactions",
        "author": "data-team@example.com",
        "created": time.time() - 30 * 86400,
        "updated": time.time() - 3600,
        "environment": "production",
        "status": "running",
        "schedule": "*/30 * * * *",
        "config": {
            "max_retries": 3,
            "timeout": 3600,
            "notifications": {
                "email": ["alerts@example.com"],
                "slack": "#ml-alerts"
            }
        },
        "steps": [
            {
                "name": "load_data",
                "type": "data_loader",
                "dependencies": [],
                "config": {"source": "s3://ops0-data/fraud/"}
            },
            {
                "name": "preprocess",
                "type": "transformer",
                "dependencies": ["load_data"],
                "config": {"scaling": "standard", "handle_missing": "impute"}
            },
            {
                "name": "train_model",
                "type": "trainer",
                "dependencies": ["preprocess"],
                "config": {"algorithm": "random_forest", "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5
                }}
            },
            {
                "name": "evaluate",
                "type": "evaluator",
                "dependencies": ["train_model"],
                "config": {"metrics": ["accuracy", "precision", "recall", "f1"]}
            }
        ],
        "resources": {
            "cpu": "2",
            "memory": "4Gi",
            "gpu": "0"
        },
        "tags": ["ml", "fraud", "production"],
        "metadata": {
            "data_sources": ["transactions_db", "user_profiles"],
            "model_type": "classification",
            "target_variable": "is_fraud",
            "features": 45,
            "training_samples": 1000000
        }
    }

    if show_code:
        # Add source code
        pipeline["source_code"] = '''import ops0
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@ops0.step
def load_data(file_path: str = "data.csv"):
    """Load transaction data."""
    data = pd.read_csv(file_path)
    return data

@ops0.step
def preprocess(data: pd.DataFrame):
    """Preprocess transaction data."""
    # Handle missing values
    data = data.fillna(data.mean())

    # Feature engineering
    data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
    data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek

    return data

@ops0.step
def train_model(data: pd.DataFrame):
    """Train fraud detection model."""
    X = data.drop(['is_fraud', 'timestamp'], axis=1)
    y = data['is_fraud']

    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X, y)

    ops0.save_model(model, "fraud-detector")
    return {"model": "fraud-detector", "features": list(X.columns)}

@ops0.step
def evaluate(model_info: dict):
    """Evaluate model performance."""
    # Load test data and model
    test_data = ops0.load("test_data")
    model = ops0.load_model(model_info["model"])

    # Calculate metrics
    predictions = model.predict(test_data)
    accuracy = (predictions == test_data['is_fraud']).mean()

    return {"accuracy": accuracy}
'''

    if show_history:
        pipeline["history"] = [
            {
                "version": "0.2.0",
                "deployed": time.time() - 3600,
                "changes": ["Updated model hyperparameters", "Added new features"],
                "deployed_by": "user@example.com"
            },
            {
                "version": "0.1.9",
                "deployed": time.time() - 86400,
                "changes": ["Fixed preprocessing bug", "Improved logging"],
                "deployed_by": "user@example.com"
            },
            {
                "version": "0.1.8",
                "deployed": time.time() - 7 * 86400,
                "changes": ["Initial production release"],
                "deployed_by": "ci@example.com"
            }
        ]

    if show_metrics:
        pipeline["metrics"] = {
            "performance": {
                "avg_runtime": 145.3,
                "success_rate": 0.98,
                "total_runs": 1456,
                "failed_runs": 29
            },
            "resource_usage": {
                "avg_cpu": "1.2 cores",
                "avg_memory": "2.8Gi",
                "peak_memory": "3.9Gi"
            },
            "model_metrics": {
                "accuracy": 0.94,
                "precision": 0.92,
                "recall": 0.88,
                "f1_score": 0.90
            }
        }

    return pipeline


def describe_step(registry: Registry, name: str, show_code: bool,
                  show_metrics: bool) -> Optional[Dict[str, Any]]:
    """Get detailed step information."""
    # Mock data for demonstration
    if name != "preprocess":
        return None

    step = {
        "name": "preprocess",
        "pipeline": "fraud-detector",
        "type": "transformer",
        "description": "Preprocess transaction data for fraud detection",
        "version": "0.2.0",
        "created": time.time() - 30 * 86400,
        "updated": time.time() - 7200,
        "inputs": [
            {"name": "data", "type": "pd.DataFrame", "required": True}
        ],
        "outputs": [
            {"name": "processed_data", "type": "pd.DataFrame"}
        ],
        "config": {
            "scaling_method": "standard",
            "handle_missing": "impute",
            "remove_outliers": True,
            "outlier_threshold": 3.0
        },
        "dependencies": {
            "python": ">=3.8",
            "packages": ["pandas>=1.5.0", "numpy>=1.23.0", "scikit-learn>=1.0.0"]
        },
        "cache": {
            "enabled": True,
            "ttl": 3600,
            "key_params": ["data.shape", "config"]
        }
    }

    if show_code:
        step["source_code"] = '''@ops0.step
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess transaction data for fraud detection.

    Args:
        data: Raw transaction data

    Returns:
        Preprocessed DataFrame ready for training
    """
    # Create copy to avoid modifying original
    df = data.copy()

    # Handle missing values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Feature engineering
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Scale numerical features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    scaling_columns = ['amount', 'balance_before', 'balance_after']
    df[scaling_columns] = scaler.fit_transform(df[scaling_columns])

    # Remove outliers
    for col in scaling_columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < 3.0]

    # Log transformation for skewed features
    df['log_amount'] = np.log1p(df['amount'].abs())

    ops0.log_metrics({
        "rows_processed": len(df),
        "features_created": len(df.columns) - len(data.columns),
        "missing_values_imputed": data.isnull().sum().sum()
    })

    return df
'''

    if show_metrics:
        step["metrics"] = {
            "performance": {
                "avg_duration": 12.3,
                "min_duration": 8.1,
                "max_duration": 23.5,
                "total_executions": 1456,
                "cache_hit_rate": 0.75
            },
            "data_stats": {
                "avg_input_rows": 50000,
                "avg_output_rows": 48500,
                "avg_features_added": 4,
                "avg_missing_imputed": 234
            },
            "resource_usage": {
                "avg_memory": "512Mi",
                "peak_memory": "890Mi",
                "avg_cpu": "0.8 cores"
            }
        }

    return step


def describe_model(registry: Registry, name: str, show_history: bool,
                   show_metrics: bool) -> Optional[Dict[str, Any]]:
    """Get detailed model information."""
    # Mock data for demonstration
    if name != "fraud-detector-rf":
        return None

    model = {
        "name": "fraud-detector-rf",
        "version": "0.2.0",
        "type": "RandomForestClassifier",
        "description": "Random Forest model for fraud detection",
        "created": time.time() - 3600,
        "created_by": "train_model@fraud-detector",
        "size": 15 * 1024 * 1024,  # 15 MB
        "location": "s3://ops0-models/fraud-detector-rf-0.2.0.pkl",
        "framework": "scikit-learn",
        "framework_version": "1.3.0",
        "algorithm": {
            "type": "RandomForestClassifier",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "random_state": 42
            }
        },
        "training_info": {
            "dataset": "fraud_transactions_v3",
            "samples": 1000000,
            "features": 45,
            "target": "is_fraud",
            "class_distribution": {"0": 0.98, "1": 0.02},
            "training_duration": 145.7,
            "trained_on": "gpu-node-01"
        },
        "input_schema": {
            "features": [
                "amount", "balance_before", "balance_after", "hour",
                "day_of_week", "merchant_category", "user_age", "account_age"
            ],
            "dtypes": {
                "amount": "float64",
                "balance_before": "float64",
                "balance_after": "float64",
                "hour": "int64",
                "day_of_week": "int64",
                "merchant_category": "category",
                "user_age": "int64",
                "account_age": "int64"
            }
        },
        "tags": ["production", "fraud", "classification"],
        "metadata": {
            "git_commit": "a1b2c3d4",
            "experiment_id": "exp_20240806_001",
            "mlflow_run_id": "run_12345"
        }
    }

    if show_history:
        model["version_history"] = [
            {
                "version": "0.2.0",
                "created": time.time() - 3600,
                "metrics": {"accuracy": 0.94, "f1": 0.90},
                "changes": "Improved feature engineering"
            },
            {
                "version": "0.1.9",
                "created": time.time() - 86400,
                "metrics": {"accuracy": 0.92, "f1": 0.87},
                "changes": "Updated hyperparameters"
            },
            {
                "version": "0.1.0",
                "created": time.time() - 30 * 86400,
                "metrics": {"accuracy": 0.89, "f1": 0.83},
                "changes": "Initial version"
            }
        ]

    if show_metrics:
        model["performance_metrics"] = {
            "classification_metrics": {
                "accuracy": 0.94,
                "precision": 0.92,
                "recall": 0.88,
                "f1_score": 0.90,
                "auc_roc": 0.96,
                "confusion_matrix": {
                    "true_negative": 9604,
                    "false_positive": 196,
                    "false_negative": 24,
                    "true_positive": 176
                }
            },
            "inference_performance": {
                "avg_latency_ms": 12.5,
                "p95_latency_ms": 23.1,
                "p99_latency_ms": 45.2,
                "throughput_qps": 850
            },
            "feature_importance": {
                "amount": 0.35,
                "balance_diff": 0.22,
                "hour": 0.15,
                "merchant_category": 0.12,
                "user_age": 0.08,
                "others": 0.08
            }
        }

    return model


def describe_deployment(registry: Registry, name: str, show_history: bool,
                        show_metrics: bool) -> Optional[Dict[str, Any]]:
    """Get detailed deployment information."""
    # Mock data for demonstration
    if name != "fraud-detector-prod":
        return None

    deployment = {
        "name": "fraud-detector-prod",
        "pipeline": "fraud-detector",
        "version": "0.2.0",
        "environment": "production",
        "status": "healthy",
        "created": time.time() - 3600,
        "deployed_by": "user@example.com",
        "deployment_config": {
            "replicas": 3,
            "strategy": "RollingUpdate",
            "max_surge": 1,
            "max_unavailable": 0
        },
        "resources": {
            "requests": {"cpu": "1", "memory": "2Gi"},
            "limits": {"cpu": "2", "memory": "4Gi"}
        },
        "endpoints": {
            "api": "https://fraud-detector.ops0.xyz",
            "health": "https://fraud-detector.ops0.xyz/health",
            "metrics": "https://fraud-detector.ops0.xyz/metrics"
        },
        "networking": {
            "service_type": "LoadBalancer",
            "ingress": {
                "enabled": True,
                "host": "fraud-detector.ops0.xyz",
                "tls": True,
                "cert_issuer": "letsencrypt"
            }
        },
        "autoscaling": {
            "enabled": True,
            "min_replicas": 2,
            "max_replicas": 10,
            "target_cpu": 70,
            "target_memory": 80
        },
        "monitoring": {
            "prometheus": True,
            "grafana_dashboard": "https://grafana.ops0.xyz/d/fraud-detector",
            "alerts": [
                {
                    "name": "high_error_rate",
                    "condition": "error_rate > 0.05",
                    "severity": "warning"
                },
                {
                    "name": "low_success_rate",
                    "condition": "success_rate < 0.95",
                    "severity": "critical"
                }
            ]
        },
        "runtime_info": {
            "container_image": "ops0/fraud-detector:0.2.0",
            "python_version": "3.9.16",
            "base_image": "python:3.9-slim",
            "last_restart": time.time() - 3600
        }
    }

    if show_history:
        deployment["deployment_history"] = [
            {
                "version": "0.2.0",
                "deployed": time.time() - 3600,
                "status": "success",
                "duration": 125,
                "deployed_by": "user@example.com"
            },
            {
                "version": "0.1.9",
                "deployed": time.time() - 86400,
                "status": "success",
                "duration": 98,
                "deployed_by": "user@example.com"
            },
            {
                "version": "0.1.8",
                "deployed": time.time() - 7 * 86400,
                "status": "rollback",
                "duration": 45,
                "deployed_by": "ci@example.com",
                "reason": "High error rate detected"
            }
        ]

    if show_metrics:
        deployment["current_metrics"] = {
            "health": {
                "status": "healthy",
                "ready_replicas": 3,
                "available_replicas": 3,
                "uptime": format_duration(time.time() - deployment["created"])
            },
            "performance": {
                "requests_per_second": 245.3,
                "avg_latency_ms": 34.2,
                "p95_latency_ms": 67.8,
                "p99_latency_ms": 125.4,
                "error_rate": 0.002
            },
            "resource_usage": {
                "cpu_usage": "45%",
                "memory_usage": "2.1Gi / 4Gi",
                "network_in": "125 MB/s",
                "network_out": "89 MB/s"
            },
            "business_metrics": {
                "predictions_today": 125430,
                "fraud_detected": 2508,
                "false_positives": 125,
                "processing_value": "$12.5M"
            }
        }

    return deployment


def display_resource_rich(resource_type: str, resource: Dict[str, Any],
                          show_code: bool, show_history: bool, show_metrics: bool):
    """Display resource with rich formatting."""
    # Main info panel
    info_lines = format_resource_info(resource_type, resource)

    title = f"{resource_type.capitalize()}: {resource['name']}"
    console.print(Panel(
        "\n".join(info_lines),
        title=f"[bold blue]{title}[/bold blue]",
        border_style="blue"
    ))

    # Additional sections based on resource type
    if resource_type in ["pipeline", "pipelines"]:
        display_pipeline_details(resource, show_code, show_history, show_metrics)
    elif resource_type in ["step", "steps"]:
        display_step_details(resource, show_code, show_metrics)
    elif resource_type in ["model", "models"]:
        display_model_details(resource, show_history, show_metrics)
    elif resource_type in ["deployment", "deployments"]:
        display_deployment_details(resource, show_history, show_metrics)


def format_resource_info(resource_type: str, resource: Dict[str, Any]) -> List[str]:
    """Format basic resource information."""
    lines = []

    # Common fields
    if 'version' in resource:
        lines.append(f"[bold]Version:[/bold] {resource['version']}")
    if 'description' in resource:
        lines.append(f"[bold]Description:[/bold] {resource['description']}")
    if 'created' in resource:
        lines.append(f"[bold]Created:[/bold] {format_time_ago(resource['created'])}")
    if 'updated' in resource:
        lines.append(f"[bold]Updated:[/bold] {format_time_ago(resource['updated'])}")
    if 'status' in resource:
        status_color = "green" if resource['status'] in ["running", "healthy"] else "yellow"
        lines.append(f"[bold]Status:[/bold] [{status_color}]{resource['status']}[/{status_color}]")

    # Type-specific fields
    if resource_type == "pipeline":
        lines.append(f"[bold]Environment:[/bold] {resource.get('environment', 'N/A')}")
        lines.append(f"[bold]Schedule:[/bold] {resource.get('schedule', 'manual')}")
        lines.append(f"[bold]Steps:[/bold] {len(resource.get('steps', []))}")
    elif resource_type == "step":
        lines.append(f"[bold]Pipeline:[/bold] {resource.get('pipeline', 'N/A')}")
        lines.append(f"[bold]Type:[/bold] {resource.get('type', 'N/A')}")
    elif resource_type == "model":
        lines.append(f"[bold]Type:[/bold] {resource.get('type', 'N/A')}")
        lines.append(f"[bold]Size:[/bold] {format_bytes(resource.get('size', 0))}")
        lines.append(f"[bold]Framework:[/bold] {resource.get('framework', 'N/A')}")
    elif resource_type == "deployment":
        lines.append(f"[bold]Pipeline:[/bold] {resource.get('pipeline', 'N/A')}")
        lines.append(f"[bold]Environment:[/bold] {resource.get('environment', 'N/A')}")
        lines.append(f"[bold]Replicas:[/bold] {resource.get('deployment_config', {}).get('replicas', 'N/A')}")

    if 'tags' in resource:
        lines.append(f"[bold]Tags:[/bold] {', '.join(resource['tags'])}")

    return lines


def display_pipeline_details(pipeline: Dict[str, Any], show_code: bool,
                             show_history: bool, show_metrics: bool):
    """Display additional pipeline details."""
    # Pipeline DAG
    if 'steps' in pipeline:
        console.print("\n[bold]Pipeline DAG:[/bold]")
        tree = Tree("📊 " + pipeline['name'])

        step_nodes = {}
        for step in pipeline['steps']:
            node = tree.add(f"📦 {step['name']} ({step['type']})")
            step_nodes[step['name']] = node

            if 'config' in step:
                for key, value in step['config'].items():
                    node.add(f"⚙️  {key}: {value}")

        console.print(tree)

    # Configuration
    if 'config' in pipeline:
        console.print("\n[bold]Configuration:[/bold]")
        console.print(Syntax(
            yaml.dump(pipeline['config'], default_flow_style=False),
            "yaml",
            theme="monokai"
        ))

    # Source code
    if show_code and 'source_code' in pipeline:
        console.print("\n[bold]Source Code:[/bold]")
        console.print(Syntax(pipeline['source_code'], "python", theme="monokai"))

    # History
    if show_history and 'history' in pipeline:
        console.print("\n[bold]Deployment History:[/bold]")
        history_table = Table(box=box.SIMPLE)
        history_table.add_column("Version", style="cyan")
        history_table.add_column("Deployed")
        history_table.add_column("Changes")
        history_table.add_column("By")

        for entry in pipeline['history']:
            history_table.add_row(
                entry['version'],
                format_time_ago(entry['deployed']),
                "\n".join(entry['changes']),
                entry['deployed_by']
            )

        console.print(history_table)

    # Metrics
    if show_metrics and 'metrics' in pipeline:
        console.print("\n[bold]Performance Metrics:[/bold]")

        # Performance table
        perf = pipeline['metrics']['performance']
        perf_table = Table(box=box.SIMPLE, show_header=False)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value")

        perf_table.add_row("Average Runtime", format_duration(perf['avg_runtime']))
        perf_table.add_row("Success Rate", f"{perf['success_rate']:.1%}")
        perf_table.add_row("Total Runs", str(perf['total_runs']))
        perf_table.add_row("Failed Runs", str(perf['failed_runs']))

        console.print(perf_table)

        # Model metrics
        if 'model_metrics' in pipeline['metrics']:
            console.print("\n[bold]Model Metrics:[/bold]")
            model_table = Table(box=box.SIMPLE, show_header=False)
            model_table.add_column("Metric", style="cyan")
            model_table.add_column("Value")

            for metric, value in pipeline['metrics']['model_metrics'].items():
                model_table.add_row(metric.replace('_', ' ').title(), f"{value:.3f}")

            console.print(model_table)


def display_step_details(step: Dict[str, Any], show_code: bool, show_metrics: bool):
    """Display additional step details."""
    # Input/Output schema
    console.print("\n[bold]Input/Output Schema:[/bold]")

    io_table = Table(box=box.SIMPLE)
    io_table.add_column("Direction", style="cyan")
    io_table.add_column("Name")
    io_table.add_column("Type")
    io_table.add_column("Required")

    for input_spec in step.get('inputs', []):
        io_table.add_row(
            "Input",
            input_spec['name'],
            input_spec['type'],
            "✓" if input_spec.get('required', True) else "✗"
        )

    for output_spec in step.get('outputs', []):
        io_table.add_row(
            "Output",
            output_spec['name'],
            output_spec['type'],
            "✓"
        )

    console.print(io_table)

    # Configuration
    if 'config' in step:
        console.print("\n[bold]Configuration:[/bold]")
        console.print(Syntax(
            yaml.dump(step['config'], default_flow_style=False),
            "yaml",
            theme="monokai"
        ))

    # Dependencies
    if 'dependencies' in step:
        console.print("\n[bold]Dependencies:[/bold]")
        console.print(f"Python: {step['dependencies'].get('python', 'any')}")
        console.print("Packages:")
        for pkg in step['dependencies'].get('packages', []):
            console.print(f"  • {pkg}")

    # Source code
    if show_code and 'source_code' in step:
        console.print("\n[bold]Source Code:[/bold]")
        console.print(Syntax(step['source_code'], "python", theme="monokai", line_numbers=True))

    # Metrics
    if show_metrics and 'metrics' in step:
        console.print("\n[bold]Performance Metrics:[/bold]")

        perf = step['metrics']['performance']
        perf_table = Table(box=box.SIMPLE, show_header=False)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value")

        perf_table.add_row("Average Duration", format_duration(perf['avg_duration']))
        perf_table.add_row("Min Duration", format_duration(perf['min_duration']))
        perf_table.add_row("Max Duration", format_duration(perf['max_duration']))
        perf_table.add_row("Cache Hit Rate", f"{perf['cache_hit_rate']:.1%}")
        perf_table.add_row("Total Executions", str(perf['total_executions']))

        console.print(perf_table)


def display_model_details(model: Dict[str, Any], show_history: bool, show_metrics: bool):
    """Display additional model details."""
    # Algorithm details
    if 'algorithm' in model:
        console.print("\n[bold]Algorithm Details:[/bold]")
        console.print(f"Type: {model['algorithm']['type']}")
        console.print("\nHyperparameters:")

        hp_table = Table(box=box.SIMPLE, show_header=False)
        hp_table.add_column("Parameter", style="cyan")
        hp_table.add_column("Value")

        for param, value in model['algorithm']['hyperparameters'].items():
            hp_table.add_row(param, str(value))

        console.print(hp_table)

    # Training info
    if 'training_info' in model:
        console.print("\n[bold]Training Information:[/bold]")
        info = model['training_info']

        train_table = Table(box=box.SIMPLE, show_header=False)
        train_table.add_column("Property", style="cyan")
        train_table.add_column("Value")

        train_table.add_row("Dataset", info['dataset'])
        train_table.add_row("Samples", f"{info['samples']:,}")
        train_table.add_row("Features", str(info['features']))
        train_table.add_row("Target", info['target'])
        train_table.add_row("Training Duration", format_duration(info['training_duration']))

        console.print(train_table)

        # Class distribution
        if 'class_distribution' in info:
            console.print("\nClass Distribution:")
            for cls, pct in info['class_distribution'].items():
                console.print(f"  • Class {cls}: {pct:.1%}")

    # Input schema
    if 'input_schema' in model:
        console.print("\n[bold]Input Schema:[/bold]")
        console.print(f"Features: {', '.join(model['input_schema']['features'][:5])}")
        if len(model['input_schema']['features']) > 5:
            console.print(f"  ... and {len(model['input_schema']['features']) - 5} more")

    # Version history
    if show_history and 'version_history' in model:
        console.print("\n[bold]Version History:[/bold]")

        version_table = Table(box=box.SIMPLE)
        version_table.add_column("Version", style="cyan")
        version_table.add_column("Created")
        version_table.add_column("Metrics")
        version_table.add_column("Changes")

        for v in model['version_history']:
            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in v['metrics'].items())
            version_table.add_row(
                v['version'],
                format_time_ago(v['created']),
                metrics_str,
                v['changes']
            )

        console.print(version_table)

    # Performance metrics
    if show_metrics and 'performance_metrics' in model:
        metrics = model['performance_metrics']

        # Classification metrics
        if 'classification_metrics' in metrics:
            console.print("\n[bold]Classification Metrics:[/bold]")
            cls_metrics = metrics['classification_metrics']

            metrics_table = Table(box=box.SIMPLE, show_header=False)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value")

            metrics_table.add_row("Accuracy", f"{cls_metrics['accuracy']:.3f}")
            metrics_table.add_row("Precision", f"{cls_metrics['precision']:.3f}")
            metrics_table.add_row("Recall", f"{cls_metrics['recall']:.3f}")
            metrics_table.add_row("F1 Score", f"{cls_metrics['f1_score']:.3f}")
            metrics_table.add_row("AUC-ROC", f"{cls_metrics['auc_roc']:.3f}")

            console.print(metrics_table)

            # Confusion matrix
            if 'confusion_matrix' in cls_metrics:
                cm = cls_metrics['confusion_matrix']
                console.print("\nConfusion Matrix:")
                console.print(f"  TN: {cm['true_negative']:,}  FP: {cm['false_positive']:,}")
                console.print(f"  FN: {cm['false_negative']:,}  TP: {cm['true_positive']:,}")

        # Feature importance
        if 'feature_importance' in metrics:
            console.print("\n[bold]Feature Importance:[/bold]")

            importance_table = Table(box=box.SIMPLE)
            importance_table.add_column("Feature", style="cyan")
            importance_table.add_column("Importance")
            importance_table.add_column("", style="blue")

            for feature, importance in sorted(
                    metrics['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
            ):
                bar = "█" * int(importance * 20)
                importance_table.add_row(feature, f"{importance:.3f}", bar)

            console.print(importance_table)


def display_deployment_details(deployment: Dict[str, Any], show_history: bool, show_metrics: bool):
    """Display additional deployment details."""
    # Deployment configuration
    if 'deployment_config' in deployment:
        console.print("\n[bold]Deployment Configuration:[/bold]")
        config = deployment['deployment_config']

        config_table = Table(box=box.SIMPLE, show_header=False)
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value")

        for key, value in config.items():
            config_table.add_row(key.replace('_', ' ').title(), str(value))

        console.print(config_table)

    # Endpoints
    if 'endpoints' in deployment:
        console.print("\n[bold]Endpoints:[/bold]")
        for name, url in deployment['endpoints'].items():
            console.print(f"  • {name}: [link]{url}[/link]")

    # Autoscaling
    if 'autoscaling' in deployment and deployment['autoscaling']['enabled']:
        console.print("\n[bold]Autoscaling:[/bold]")
        auto = deployment['autoscaling']
        console.print(f"  • Replicas: {auto['min_replicas']} - {auto['max_replicas']}")
        console.print(f"  • Target CPU: {auto['target_cpu']}%")
        console.print(f"  • Target Memory: {auto['target_memory']}%")

    # Monitoring
    if 'monitoring' in deployment:
        console.print("\n[bold]Monitoring:[/bold]")
        mon = deployment['monitoring']

        if mon.get('grafana_dashboard'):
            console.print(f"  • Dashboard: [link]{mon['grafana_dashboard']}[/link]")

        if 'alerts' in mon:
            console.print("  • Alerts:")
            for alert in mon['alerts']:
                severity_color = "red" if alert['severity'] == "critical" else "yellow"
                console.print(
                    f"    - {alert['name']}: {alert['condition']} "
                    f"[{severity_color}]({alert['severity']})[/{severity_color}]"
                )

    # Deployment history
    if show_history and 'deployment_history' in deployment:
        console.print("\n[bold]Deployment History:[/bold]")

        history_table = Table(box=box.SIMPLE)
        history_table.add_column("Version", style="cyan")
        history_table.add_column("Deployed")
        history_table.add_column("Status")
        history_table.add_column("Duration")
        history_table.add_column("By")

        for entry in deployment['deployment_history']:
            status_color = "green" if entry['status'] == "success" else "red"
            history_table.add_row(
                entry['version'],
                format_time_ago(entry['deployed']),
                f"[{status_color}]{entry['status']}[/{status_color}]",
                f"{entry['duration']}s",
                entry['deployed_by']
            )

        console.print(history_table)

    # Current metrics
    if show_metrics and 'current_metrics' in deployment:
        metrics = deployment['current_metrics']

        # Health metrics
        if 'health' in metrics:
            console.print("\n[bold]Health Status:[/bold]")
            health = metrics['health']

            health_table = Table(box=box.SIMPLE, show_header=False)
            health_table.add_column("Metric", style="cyan")
            health_table.add_column("Value")

            status_color = "green" if health['status'] == "healthy" else "red"
            health_table.add_row("Status", f"[{status_color}]{health['status']}[/{status_color}]")
            health_table.add_row("Ready Replicas", f"{health['ready_replicas']}/{health['available_replicas']}")
            health_table.add_row("Uptime", health['uptime'])

            console.print(health_table)

        # Performance metrics
        if 'performance' in metrics:
            console.print("\n[bold]Performance:[/bold]")
            perf = metrics['performance']

            perf_table = Table(box=box.SIMPLE, show_header=False)
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Value")

            perf_table.add_row("Requests/sec", f"{perf['requests_per_second']:.1f}")
            perf_table.add_row("Avg Latency", f"{perf['avg_latency_ms']:.1f} ms")
            perf_table.add_row("P95 Latency", f"{perf['p95_latency_ms']:.1f} ms")
            perf_table.add_row("P99 Latency", f"{perf['p99_latency_ms']:.1f} ms")
            perf_table.add_row("Error Rate", f"{perf['error_rate']:.3%}")

            console.print(perf_table)

        # Resource usage
        if 'resource_usage' in metrics:
            console.print("\n[bold]Resource Usage:[/bold]")
            resources = metrics['resource_usage']

            for resource, usage in resources.items():
                console.print(f"  • {resource.replace('_', ' ').title()}: {usage}")

        # Business metrics
        if 'business_metrics' in metrics:
            console.print("\n[bold]Business Metrics:[/bold]")
            biz = metrics['business_metrics']

            biz_table = Table(box=box.SIMPLE, show_header=False)
            biz_table.add_column("Metric", style="cyan")
            biz_table.add_column("Value")

            for metric, value in biz.items():
                formatted_metric = metric.replace('_', ' ').title()
                if isinstance(value, (int, float)) and value > 1000:
                    formatted_value = f"{value:,}"
                else:
                    formatted_value = str(value)
                biz_table.add_row(formatted_metric, formatted_value)

            console.print(biz_table)


class Registry:
    """Mock registry for demonstration."""

    def __init__(self, project_root: Path):
        self.project_root = project_root