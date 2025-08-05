"""
ops0 CLI Commands Implementation

Core command implementations for the ops0 CLI with enhanced functionality.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from .utils import (
    console,
    print_success,
    print_error,
    print_info,
    print_warning,
    confirm_action,
    show_pipeline_tree,
    ProgressTracker,
    ensure_directory,
    show_code_snippet,
)

# Handle imports for both development and production
try:
    from ..core.graph import PipelineGraph
    from ..core.executor import run, deploy
    from ..core.storage import storage
    from ..runtime.containers import container_orchestrator
except ImportError:
    # Development mode
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    try:
        from .core.graph import PipelineGraph
        from .core.executor import run, deploy
        from .core.storage import storage
        from .runtime.containers import container_orchestrator
    except ImportError:
        # Graceful fallback
        PipelineGraph = None
        run = None
        deploy = None
        storage = None
        container_orchestrator = None

# Project templates
PROJECT_TEMPLATES = {
    "basic": {
        "description": "Basic ops0 pipeline with simple data processing",
        "files": ["pipeline.py", "requirements.txt", "README.md"]
    },
    "ml": {
        "description": "Machine learning pipeline with training and prediction",
        "files": ["pipeline.py", "train.py", "predict.py", "requirements.txt", "README.md", "data/sample.csv"]
    },
    "advanced": {
        "description": "Advanced pipeline with monitoring and custom integrations",
        "files": ["pipeline.py", "monitoring.py", "integrations.py", "requirements.txt", "README.md", "ops0.toml"]
    }
}


def create_project(
        name: str,
        template: str = "basic",
        force: bool = False,
        dry_run: bool = False
) -> List[Tuple[Path, str]]:
    """
    Create a new ops0 project with specified template.

    Args:
        name: Project name
        template: Template type (basic, ml, advanced)
        force: Overwrite existing files
        dry_run: Show what would be created without creating

    Returns:
        List of (file_path, description) tuples
    """
    if template not in PROJECT_TEMPLATES:
        available = ", ".join(PROJECT_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template}'. Available: {available}")

    project_path = Path(name)

    # Check if directory exists
    if project_path.exists() and not force:
        if any(project_path.iterdir()):
            raise ValueError(f"Directory '{name}' already exists and is not empty. Use --force to overwrite.")

    template_info = PROJECT_TEMPLATES[template]
    created_files = []

    with ProgressTracker(
            ["Creating structure", "Generating files", "Setting up config"],
            "Creating Project"
    ) as progress:

        # Create project directory
        if not dry_run:
            ensure_directory(project_path)
            ensure_directory(project_path / ".ops0" / "storage")
            ensure_directory(project_path / "logs")

        created_files.append((project_path, "Project directory"))
        created_files.append((project_path / ".ops0", "ops0 metadata directory"))

        progress.next_step("Generating files")

        # Generate files based on template
        for file_name in template_info["files"]:
            file_path = project_path / file_name

            if "/" in file_name:  # Handle subdirectories
                file_path.parent.mkdir(parents=True, exist_ok=True)

            content = _generate_file_content(file_name, template, name)

            if not dry_run:
                with open(file_path, 'w') as f:
                    f.write(content)

            created_files.append((file_path, _get_file_description(file_name)))

        progress.next_step("Setting up config")

        # Create additional config files
        if template == "advanced":
            # Create ops0.toml
            config_content = _generate_config_file(name)
            config_path = project_path / "ops0.toml"

            if not dry_run:
                with open(config_path, 'w') as f:
                    f.write(config_content)

            created_files.append((config_path, "ops0 configuration"))

        # Create .gitignore
        gitignore_content = _generate_gitignore()
        gitignore_path = project_path / ".gitignore"

        if not dry_run:
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content)

        created_files.append((gitignore_path, "Git ignore file"))

        progress.complete(f"Project '{name}' created!")

    return created_files


def _generate_file_content(file_name: str, template: str, project_name: str) -> str:
    """Generate content for a specific file based on template"""

    if file_name == "pipeline.py":
        if template == "basic":
            return f'''"""
{project_name} - Basic ops0 Pipeline

A simple data processing pipeline using ops0.
"""

import ops0
import pandas as pd
from datetime import datetime

@ops0.step
def load_data():
    """Load sample data for processing"""
    # In a real pipeline, this would load from your data source
    data = pd.DataFrame({{
        'id': range(1, 101),
        'value': [i * 2 for i in range(1, 101)],
        'category': ['A', 'B', 'C'] * 33 + ['A'],
        'timestamp': [datetime.now()] * 100
    }})

    ops0.storage.save("raw_data", data)
    print(f"✓ Loaded {{len(data)}} rows of data")
    return data

@ops0.step
def process_data():
    """Process and clean the data"""
    data = ops0.storage.load("raw_data")

    # Simple processing
    processed = data.copy()
    processed['value_normalized'] = processed['value'] / processed['value'].max()
    processed['is_high_value'] = processed['value'] > processed['value'].median()

    ops0.storage.save("processed_data", processed)
    print(f"✓ Processed {{len(processed)}} rows")
    return processed

@ops0.step
def analyze_results():
    """Analyze the processed data"""
    data = ops0.storage.load("processed_data")

    analysis = {{
        'total_rows': len(data),
        'high_value_count': data['is_high_value'].sum(),
        'average_value': data['value'].mean(),
        'categories': data['category'].value_counts().to_dict()
    }}

    ops0.storage.save("analysis_results", analysis)
    print(f"✓ Analysis complete: {{analysis['high_value_count']}} high-value items found")
    return analysis

if __name__ == "__main__":
    print("🚀 Running {{project_name}} pipeline...")

    with ops0.pipeline("{project_name}"):
        # Define pipeline steps
        load_data()
        process_data()
        analyze_results()

        # Execute locally
        print("\\n📊 Executing pipeline...")
        results = ops0.run(mode="local")

        print("\\n✅ Pipeline completed successfully!")
        print(f"Results available in storage: {{list(results.keys())}}")

        # To deploy to production:
        # ops0.deploy()
'''

        elif template == "ml":
            return f'''"""
{project_name} - ML Pipeline

A machine learning pipeline with training and prediction using ops0.
"""

import ops0
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

@ops0.step
def load_training_data():
    """Load and prepare training data"""
    # Generate sample training data
    np.random.seed(42)
    n_samples = 1000

    data = pd.DataFrame({{
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.uniform(0, 1, n_samples)
    }})

    # Create target based on features
    data['target'] = (
        (data['feature1'] > 0) & 
        (data['feature2'] > data['feature3'])
    ).astype(int)

    ops0.storage.save("training_data", data)
    print(f"✓ Loaded {{len(data)}} training samples")
    return data

@ops0.step
def preprocess_features():
    """Feature engineering and preprocessing"""
    data = ops0.storage.load("training_data")

    # Feature engineering
    processed = data.copy()
    processed['feature1_squared'] = processed['feature1'] ** 2
    processed['feature_interaction'] = processed['feature1'] * processed['feature2']
    processed['feature_ratio'] = processed['feature3'] / (processed['feature4'] + 1e-8)

    # Split features and target
    feature_cols = [col for col in processed.columns if col != 'target']
    X = processed[feature_cols]
    y = processed['target']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    ops0.storage.save("X_train", X_train)
    ops0.storage.save("X_test", X_test)
    ops0.storage.save("y_train", y_train)
    ops0.storage.save("y_test", y_test)

    print(f"✓ Preprocessed features: {{len(feature_cols)}} features")
    print(f"✓ Train set: {{len(X_train)}} samples, Test set: {{len(X_test)}} samples")

    return {{"n_features": len(feature_cols), "train_size": len(X_train)}}

@ops0.step
def train_model():
    """Train the machine learning model"""
    X_train = ops0.storage.load("X_train")
    y_train = ops0.storage.load("y_train")

    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    print("🤖 Training model...")
    model.fit(X_train, y_train)

    # Feature importance
    feature_importance = dict(zip(X_train.columns, model.feature_importances_))

    ops0.storage.save("trained_model", model)
    ops0.storage.save("feature_importance", feature_importance)

    print("✓ Model training complete")
    return {{"model_type": "RandomForest", "n_estimators": 100}}

@ops0.step
def evaluate_model():
    """Evaluate model performance"""
    model = ops0.storage.load("trained_model")
    X_test = ops0.storage.load("X_test")
    y_test = ops0.storage.load("y_test")

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    evaluation_results = {{
        'accuracy': accuracy,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'test_samples': len(y_test)
    }}

    ops0.storage.save("evaluation_results", evaluation_results)
    ops0.storage.save("predictions", y_pred)
    ops0.storage.save("prediction_probabilities", y_proba)

    print(f"✓ Model evaluation complete")
    print(f"  Accuracy: {{accuracy:.3f}}")
    print(f"  F1-Score: {{evaluation_results['f1_score']:.3f}}")

    return evaluation_results

@ops0.step
def generate_model_report():
    """Generate comprehensive model report"""
    evaluation = ops0.storage.load("evaluation_results")
    feature_importance = ops0.storage.load("feature_importance")

    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )

    report = {{
        'model_performance': evaluation,
        'top_features': sorted_features[:5],
        'model_ready_for_production': evaluation['accuracy'] > 0.7,
        'recommendations': []
    }}

    # Add recommendations
    if evaluation['accuracy'] < 0.8:
        report['recommendations'].append("Consider feature engineering")
    if evaluation['precision'] < 0.8:
        report['recommendations'].append("Review precision-recall tradeoff")

    ops0.storage.save("model_report", report)

    print("✓ Model report generated")
    if report['model_ready_for_production']:
        print("🎉 Model ready for production deployment!")
    else:
        print("⚠️  Model needs improvement before production")

    return report

if __name__ == "__main__":
    print("🚀 Running {{project_name}} ML pipeline...")

    with ops0.pipeline("{project_name}-ml"):
        # Define ML pipeline steps
        load_training_data()
        preprocess_features()
        train_model()
        evaluate_model()
        generate_model_report()

        # Execute locally
        print("\\n📊 Executing ML pipeline...")
        results = ops0.run(mode="local")

        print("\\n✅ ML Pipeline completed successfully!")

        # Show final results
        if "generate_model_report" in results:
            report = results["generate_model_report"]
            print(f"\\n📊 Final Model Performance:")
            print(f"  Accuracy: {{report['model_performance']['accuracy']:.3f}}")
            print(f"  Ready for production: {{report['model_ready_for_production']}}")

        # To deploy to production:
        # print("\\n🚀 Deploying to production...")
        # ops0.deploy()
'''

        elif template == "advanced":
            return f'''"""
{project_name} - Advanced ops0 Pipeline

Advanced pipeline with monitoring, error handling, and custom integrations.
"""

import ops0
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ops0.step
def extract_data(source: str = "api"):
    """Extract data from various sources with error handling"""
    try:
        if source == "api":
            # Simulate API data extraction
            data = {{
                'records': [
                    {{'id': i, 'value': np.random.randn(), 'timestamp': datetime.now()}}
                    for i in range(1000)
                ],
                'metadata': {{
                    'source': 'api',
                    'extracted_at': datetime.now(),
                    'total_records': 1000
                }}
            }}
        elif source == "database":
            # Simulate database extraction
            data = {{
                'records': [
                    {{'id': i, 'category': f'cat_{{i%5}}', 'score': np.random.uniform(0, 100)}}
                    for i in range(500)
                ],
                'metadata': {{
                    'source': 'database',
                    'extracted_at': datetime.now(),
                    'total_records': 500
                }}
            }}
        else:
            raise ValueError(f"Unknown source: {{source}}")

        ops0.storage.save("raw_data", data)
        ops0.storage.save("extraction_metadata", data['metadata'])

        logger.info(f"✓ Extracted {{data['metadata']['total_records']}} records from {{source}}")
        return data['metadata']

    except Exception as e:
        logger.error(f"Data extraction failed: {{e}}")
        # Save error info for monitoring
        error_info = {{
            'error': str(e),
            'step': 'extract_data',
            'timestamp': datetime.now(),
            'source': source
        }}
        ops0.storage.save("extraction_error", error_info)
        raise

@ops0.step
def validate_data():
    """Validate data quality with comprehensive checks"""
    data = ops0.storage.load("raw_data")
    metadata = ops0.storage.load("extraction_metadata")

    records = pd.DataFrame(data['records'])

    validation_results = {{
        'total_records': len(records),
        'null_counts': records.isnull().sum().to_dict(),
        'duplicate_count': records.duplicated().sum(),
        'data_types': records.dtypes.astype(str).to_dict(),
        'validation_passed': True,
        'issues': []
    }}

    # Data quality checks
    if validation_results['duplicate_count'] > 0:
        validation_results['issues'].append(f"Found {{validation_results['duplicate_count']}} duplicates")

    for col, null_count in validation_results['null_counts'].items():
        if null_count > len(records) * 0.1:  # More than 10% nulls
            validation_results['issues'].append(f"High null rate in {{col}}: {{null_count/len(records):.1%}}")
            validation_results['validation_passed'] = False

    # Validate against expected schema
    if metadata['source'] == 'api':
        required_cols = ['id', 'value', 'timestamp']
    else:
        required_cols = ['id', 'category', 'score']

    missing_cols = set(required_cols) - set(records.columns)
    if missing_cols:
        validation_results['issues'].append(f"Missing required columns: {{missing_cols}}")
        validation_results['validation_passed'] = False

    ops0.storage.save("validation_results", validation_results)
    ops0.storage.save("validated_data", records)

    if validation_results['validation_passed']:
        logger.info("✓ Data validation passed")
    else:
        logger.warning(f"⚠️ Data validation issues: {{validation_results['issues']}}")

    return validation_results

@ops0.step
def transform_data():
    """Advanced data transformation with feature engineering"""
    data = ops0.storage.load("validated_data")
    validation = ops0.storage.load("validation_results")

    if not validation['validation_passed']:
        logger.warning("Proceeding with transformation despite validation issues")

    # Advanced transformations
    transformed = data.copy()

    # Handle different data sources
    if 'value' in transformed.columns:
        # API data transformations
        transformed['value_normalized'] = (
            transformed['value'] - transformed['value'].mean()
        ) / transformed['value'].std()
        transformed['value_percentile'] = transformed['value'].rank(pct=True)
        transformed['is_outlier'] = np.abs(transformed['value_normalized']) > 2

    if 'score' in transformed.columns:
        # Database data transformations
        transformed['score_category'] = pd.cut(
            transformed['score'], 
            bins=[0, 25, 50, 75, 100], 
            labels=['low', 'medium', 'high', 'very_high']
        )
        transformed['score_normalized'] = transformed['score'] / 100

    # Universal transformations
    if 'timestamp' in transformed.columns:
        transformed['hour'] = pd.to_datetime(transformed['timestamp']).dt.hour
        transformed['is_business_hours'] = transformed['hour'].between(9, 17)

    # Create summary statistics
    transformation_stats = {{
        'input_rows': len(data),
        'output_rows': len(transformed),
        'new_columns': list(set(transformed.columns) - set(data.columns)),
        'outliers_detected': transformed.get('is_outlier', pd.Series()).sum() if 'is_outlier' in transformed.columns else 0,
        'transformation_time': datetime.now()
    }}

    ops0.storage.save("transformed_data", transformed)
    ops0.storage.save("transformation_stats", transformation_stats)

    logger.info(f"✓ Data transformation complete: {{len(transformation_stats['new_columns'])}} new features")
    return transformation_stats

@ops0.step
def analyze_and_report():
    """Generate comprehensive analysis and reporting"""
    data = ops0.storage.load("transformed_data")
    transform_stats = ops0.storage.load("transformation_stats")
    validation = ops0.storage.load("validation_results")

    # Comprehensive analysis
    analysis = {{
        'data_summary': {{
            'total_records': len(data),
            'total_features': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
        }},
        'quality_metrics': {{
            'validation_passed': validation['validation_passed'],
            'completeness_score': 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
            'uniqueness_score': 1 - (data.duplicated().sum() / len(data)),
        }},
        'feature_analysis': {{}},
        'recommendations': [],
        'ready_for_next_stage': True
    }}

    # Feature-specific analysis
    for column in data.select_dtypes(include=[np.number]).columns:
        analysis['feature_analysis'][column] = {{
            'mean': float(data[column].mean()),
            'std': float(data[column].std()),
            'min': float(data[column].min()),
            'max': float(data[column].max()),
            'null_count': int(data[column].isnull().sum())
        }}

    # Generate recommendations
    if analysis['quality_metrics']['completeness_score'] < 0.95:
        analysis['recommendations'].append("Consider imputation for missing values")
        analysis['ready_for_next_stage'] = False

    if analysis['quality_metrics']['uniqueness_score'] < 0.99:
        analysis['recommendations'].append("Review and handle duplicate records")

    if len(data) < 100:
        analysis['recommendations'].append("Dataset might be too small for reliable analysis")
        analysis['ready_for_next_stage'] = False

    # Performance metrics
    analysis['performance'] = {{
        'processing_time': str(datetime.now() - transform_stats['transformation_time']),
        'throughput_records_per_second': len(data) / max((datetime.now() - transform_stats['transformation_time']).total_seconds(), 1)
    }}

    ops0.storage.save("final_analysis", analysis)

    # Log results
    logger.info("✓ Analysis complete")
    logger.info(f"  Records processed: {{analysis['data_summary']['total_records']}}")
    logger.info(f"  Data quality score: {{analysis['quality_metrics']['completeness_score']:.2%}}")
    logger.info(f"  Ready for next stage: {{analysis['ready_for_next_stage']}}")

    if analysis['recommendations']:
        logger.warning(f"  Recommendations: {{', '.join(analysis['recommendations'])}}")

    return analysis

if __name__ == "__main__":
    print("🚀 Running {{project_name}} advanced pipeline...")

    # Configure pipeline with monitoring
    with ops0.pipeline("{project_name}-advanced") as pipeline:

        # Define advanced pipeline with error handling
        try:
            # Extract data (configurable source)
            source_type = os.getenv('DATA_SOURCE', 'api')
            extract_data(source=source_type)

            # Validate and transform
            validate_data()
            transform_data()
            analyze_and_report()

            # Execute pipeline
            print("\\n📊 Executing advanced pipeline...")
            results = ops0.run(mode="local")

            # Show comprehensive results
            if "analyze_and_report" in results:
                final_analysis = results["analyze_and_report"]

                print("\\n✅ Advanced Pipeline completed successfully!")
                print(f"\\n📊 Pipeline Summary:")
                print(f"  Records: {{final_analysis['data_summary']['total_records']}}")
                print(f"  Features: {{final_analysis['data_summary']['total_features']}}")
                print(f"  Quality Score: {{final_analysis['quality_metrics']['completeness_score']:.2%}}")
                print(f"  Ready for production: {{final_analysis['ready_for_next_stage']}}")

                if final_analysis['recommendations']:
                    print(f"\\n💡 Recommendations:")
                    for rec in final_analysis['recommendations']:
                        print(f"  • {{rec}}")

            # Auto-deploy if configured and quality is good
            if (os.getenv('AUTO_DEPLOY', 'false').lower() == 'true' and 
                results.get("analyze_and_report", {{}}).get('ready_for_next_stage', False)):
                print("\\n🚀 Auto-deploying to production...")
                deployment_result = ops0.deploy()
                print(f"✅ Deployed successfully: {{deployment_result.get('url', 'Unknown URL')}}")

        except Exception as e:
            logger.error(f"Pipeline failed: {{e}}")
            print(f"\\n❌ Pipeline failed: {{e}}")

            # Save failure information for debugging
            failure_info = {{
                'error': str(e),
                'timestamp': datetime.now(),
                'pipeline': "{project_name}-advanced"
            }}

            try:
                ops0.storage.save("pipeline_failure", failure_info)
            except:
                pass  # Don't fail on failure logging

            raise
'''

    elif file_name == "requirements.txt":
        if template == "basic":
            return """ops0>=0.1.0
pandas>=2.0.0
numpy>=1.24.0
"""
        elif template == "ml":
            return """ops0>=0.1.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
"""
        elif template == "advanced":
            return """ops0>=0.1.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
requests>=2.31.0
pydantic>=2.0.0
rich>=13.0.0
watchdog>=3.0.0
"""

    elif file_name == "README.md":
        return f"""# {project_name}

{PROJECT_TEMPLATES[template]['description']}

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the pipeline locally:**
   ```bash
   python pipeline.py
   ```

3. **Deploy to production:**
   ```bash
   ops0 deploy
   ```

## Project Structure

```
{project_name}/
├── pipeline.py          # Main pipeline definition
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .ops0/              # ops0 metadata
└── logs/               # Pipeline logs
```

## ops0 Commands

- `ops0 run --local` - Run pipeline locally
- `ops0 status` - Check pipeline status
- `ops0 logs` - View pipeline logs
- `ops0 deploy` - Deploy to production

## Documentation

- [ops0 Documentation](https://docs.ops0.xyz)
- [Getting Started Guide](https://docs.ops0.xyz/getting-started)
- [API Reference](https://docs.ops0.xyz/api)

## Support

- [Discord Community](https://discord.gg/ops0)
- [GitHub Issues](https://github.com/ops0-mlops/ops0/issues)
"""

    else:
        # Handle additional files for specific templates
        if template == "ml" and file_name == "data/sample.csv":
            return """id,feature1,feature2,target
1,0.5,1.2,1
2,-0.3,0.8,0
3,1.1,-0.5,1
4,0.7,0.9,1
5,-0.8,-0.2,0
"""

        return f"# {file_name} for {project_name}\n# Generated by ops0\n"


def _get_file_description(file_name: str) -> str:
    """Get description for a file"""
    descriptions = {
        "pipeline.py": "Main pipeline definition",
        "requirements.txt": "Python dependencies",
        "README.md": "Project documentation",
        "train.py": "Model training script",
        "predict.py": "Prediction script",
        "monitoring.py": "Pipeline monitoring",
        "integrations.py": "Custom integrations",
        "ops0.toml": "ops0 configuration",
        "data/sample.csv": "Sample training data",
        ".gitignore": "Git ignore rules"
    }
    return descriptions.get(file_name, f"Generated {file_name}")


def _generate_config_file(project_name: str) -> str:
    """Generate ops0.toml configuration file"""
    return f'''[project]
name = "{project_name}"
version = "1.0.0"
description = "Advanced ops0 pipeline with monitoring"

[deployment]
environment = "production"
auto_deploy = false
deployment_timeout = 300

[monitoring]
enable_monitoring = true
alert_channels = []
metrics_retention_days = 30

[storage]
backend = "local"
path = ".ops0/storage"

[containers]
registry = "ghcr.io/ops0-mlops"
build_on_deploy = true
push_to_registry = true

[development]
hot_reload = true
debug_mode = false
'''


def _generate_gitignore() -> str:
    """Generate .gitignore file"""
    return """# Python
__pycache__/
*.py[cod]
*$py.class
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
.venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# ops0 specific
.ops0/
*.ops0
logs/
.env.local
.env.production

# Data files (add your data directories)
data/
models/
artifacts/
*.csv
*.json
*.parquet

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Testing
.coverage
.pytest_cache/
htmlcov/

# Documentation
docs/_build/
"""


def show_pipeline_status(pipeline_name: Optional[str] = None) -> Dict[str, Any]:
    """Show detailed pipeline status information"""
    if not PipelineGraph:
        return {"error": "Pipeline module not available"}

    current = PipelineGraph.get_current()
    if not current and not pipeline_name:
        return {"status": "No active pipeline"}

    if pipeline_name:
        # Load specific pipeline (would implement pipeline discovery)
        print_info(f"Pipeline '{pipeline_name}' status not implemented yet")
        return {"status": "Not implemented"}

    # Show current pipeline status
    status = {
        "name": current.name,
        "steps": len(current.steps),
        "status": "Active",
        "created": time.time(),
        "steps_detail": {}
    }

    # Add step details
    for step_name, step_node in current.steps.items():
        status["steps_detail"][step_name] = {
            "status": "Executed" if getattr(step_node, 'executed', False) else "Pending",
            "dependencies": list(step_node.dependencies),
            "result_available": step_node.result is not None
        }

    return status


def containerize_current_pipeline(push: bool = False, step: Optional[str] = None) -> Dict[str, Any]:
    """Containerize the current pipeline or specific step"""
    if not PipelineGraph or not container_orchestrator:
        raise RuntimeError("Pipeline or container modules not available")

    current = PipelineGraph.get_current()
    if not current:
        raise RuntimeError("No active pipeline found")

    if step:
        if step not in current.steps:
            available_steps = ", ".join(current.steps.keys())
            raise ValueError(f"Step '{step}' not found. Available: {available_steps}")

        print_info(f"Containerizing step: {step}")
        # Would implement single step containerization
        return {"containerized_steps": [step]}

    print_info(f"Containerizing entire pipeline: {current.name}")

    # Enable container building
    if push:
        os.environ["OPS0_BUILD_CONTAINERS"] = "true"

    try:
        specs = container_orchestrator.containerize_pipeline(current)

        result = {
            "pipeline": current.name,
            "containerized_steps": list(specs.keys()),
            "total_containers": len(specs),
            "registry_pushed": push,
            "containers": {}
        }

        for step_name, spec in specs.items():
            result["containers"][step_name] = {
                "image_tag": spec.image_tag,
                "memory_limit": spec.memory_limit,
                "gpu_support": spec.needs_gpu,
                "requirements_count": len(spec.requirements)
            }

        return result

    except Exception as e:
        print_error(f"Containerization failed: {str(e)}")
        raise


def debug_pipeline_execution(step: Optional[str] = None, interactive: bool = False) -> Dict[str, Any]:
    """Debug pipeline execution with detailed information"""
    if not PipelineGraph:
        raise RuntimeError("Pipeline module not available")

    current = PipelineGraph.get_current()
    if not current:
        raise RuntimeError("No active pipeline found")

    debug_info = {
        "pipeline": current.name,
        "total_steps": len(current.steps),
        "execution_order": current.build_execution_order(),
        "step_details": {}
    }

    if step:
        if step not in current.steps:
            available_steps = ", ".join(current.steps.keys())
            raise ValueError(f"Step '{step}' not found. Available: {available_steps}")

        step_node = current.steps[step]
        debug_info["focused_step"] = {
            "name": step,
            "dependencies": list(step_node.dependencies),
            "dependents": [name for name, node in current.steps.items()
                           if step in current.get_step_dependencies(name)],
            "executed": getattr(step_node, 'executed', False),
            "has_result": step_node.result is not None,
            "function_name": step_node.func.__name__,
            "function_file": step_node.func.__code__.co_filename,
            "function_line": step_node.func.__code__.co_firstlineno
        }

        if interactive:
            console.print("\n🐛 Interactive Debug Mode")
            console.print("Available commands: status, deps, run, inspect, help, exit")

            while True:
                command = prompt_input("debug> ").strip().lower()

                if command == "exit":
                    break
                elif command == "status":
                    console.print(f"Step: {step}")
                    console.print(f"Executed: {debug_info['focused_step']['executed']}")
                    console.print(f"Dependencies: {debug_info['focused_step']['dependencies']}")
                elif command == "deps":
                    deps = debug_info['focused_step']['dependencies']
                    if deps:
                        console.print(f"Dependencies: {', '.join(deps)}")
                    else:
                        console.print("No dependencies")
                elif command == "help":
                    console.print("Commands: status, deps, run, inspect, help, exit")
                else:
                    console.print(f"Unknown command: {command}")

    else:
        # Debug entire pipeline
        for step_name, step_node in current.steps.items():
            debug_info["step_details"][step_name] = {
                "dependencies": list(step_node.dependencies),
                "executed": getattr(step_node, 'executed', False),
                "has_result": step_node.result is not None
            }

    return debug_info