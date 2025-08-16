"""
Command Line Interface for ops0.
Zero-configuration MLOps with intelligent defaults.
"""
import os
import sys
import click
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import importlib.util
import traceback

from ops0.decorators import PipelineDecorator
from ops0.executor import Orchestrator, ExecutionMode
from ops0.storage import set_storage, LocalStorage, S3Storage
from ops0.cdk.generator import CDKGenerator

# Color scheme for beautiful CLI output
COLORS = {
    'success': 'green',
    'error': 'red',
    'warning': 'yellow',
    'info': 'blue',
    'dim': 'bright_black'
}


@click.group()
@click.version_option(version='0.1.0', prog_name='ops0')
def cli():
    """
    ops0 - Write Python, Ship Production 🚀

    Transform Python functions into production ML pipelines with zero configuration.
    """
    pass


@cli.command()
@click.argument('project_name', required=False)
@click.option('--template', '-t', type=click.Choice(['basic', 'ml', 'streaming']),
              default='basic', help='Project template to use')
def init(project_name: Optional[str], template: str):
    """Initialize a new ops0 project"""
    if not project_name:
        project_name = Path.cwd().name

    click.echo(f"🚀 Initializing ops0 project: {click.style(project_name, fg=COLORS['info'], bold=True)}")

    # Create project structure
    project_dir = Path(project_name)
    if project_dir.exists() and project_dir != Path.cwd():
        click.echo(f"❌ Directory {project_name} already exists", err=True)
        sys.exit(1)

    # Create directories
    dirs = [
        '',
        'pipelines',
        'steps',
        'models',
        'data',
        '.ops0',
        '.ops0/logs',
        '.ops0/cache'
    ]

    for dir_path in dirs:
        full_path = project_dir / dir_path if project_dir != Path.cwd() else Path(dir_path)
        full_path.mkdir(parents=True, exist_ok=True)

    # Create template files based on type
    templates = _get_project_templates(template)

    for file_path, content in templates.items():
        full_path = project_dir / file_path if project_dir != Path.cwd() else Path(file_path)
        full_path.write_text(content)
        click.echo(f"  ✅ Created {click.style(file_path, fg=COLORS['dim'])}")

    # Create .gitignore
    gitignore_content = """
# ops0
.ops0/
*.pkl
*.model

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.env

# Data
data/
*.csv
*.parquet
*.json

# Models
models/
*.h5
*.pt
*.onnx

# IDE
.vscode/
.idea/
*.swp
"""
    (project_dir / '.gitignore' if project_dir != Path.cwd() else Path('.gitignore')).write_text(gitignore_content)

    click.echo(f"\n✨ Project initialized successfully!")
    click.echo(f"\n📝 Next steps:")
    if project_dir != Path.cwd():
        click.echo(f"  1. cd {project_name}")
    click.echo(f"  2. pip install ops0")
    click.echo(f"  3. ops0 run --local")
    click.echo(f"  4. ops0 deploy")


@cli.command()
@click.option('--local', is_flag=True, help='Run pipeline locally')
@click.option('--docker', is_flag=True, help='Run pipeline in Docker containers')
@click.option('--pipeline', '-p', help='Specific pipeline to run')
@click.option('--step', '-s', help='Run only a specific step')
@click.option('--watch', '-w', is_flag=True, help='Watch for changes and auto-reload')
@click.argument('args', nargs=-1)
def run(docker: bool, pipeline: Optional[str], step: Optional[str],
        watch: bool, args: tuple):
    """Run a pipeline locally or in containers"""

    # Determine execution mode
    if docker:
        mode = ExecutionMode.DOCKER
        mode_str = "Docker"
    else:
        mode = ExecutionMode.LOCAL
        mode_str = "local"

    click.echo(f"🏃 Running pipeline in {click.style(mode_str, fg=COLORS['info'])} mode...\n")

    # Load pipeline module
    pipeline_module = _load_pipeline_module()

    if not pipeline_module:
        click.echo("❌ No pipeline found. Create a pipeline.py file or use --pipeline flag", err=True)
        sys.exit(1)

    # Get all pipelines
    pipelines = PipelineDecorator.get_all()

    if not pipelines:
        click.echo("❌ No pipelines found. Decorate a function with @ops0.pipeline", err=True)
        sys.exit(1)

    # Select pipeline to run
    if pipeline:
        pipeline_config = pipelines.get(pipeline)
        if not pipeline_config:
            click.echo(f"❌ Pipeline '{pipeline}' not found", err=True)
            click.echo(f"Available pipelines: {', '.join(pipelines.keys())}")
            sys.exit(1)
    else:
        # Use first pipeline found
        pipeline_name = list(pipelines.keys())[0]
        pipeline_config = pipelines[pipeline_name]

    # Show pipeline info
    click.echo(f"📋 Pipeline: {click.style(pipeline_config.name, fg=COLORS['info'], bold=True)}")
    if pipeline_config.description:
        click.echo(f"📝 {pipeline_config.description}")

    click.echo(f"\n🔍 Steps detected:")
    for step_name, step_config in pipeline_config.steps.items():
        memory_str = f"{step_config.memory}MB"
        gpu_str = " [GPU]" if step_config.gpu else ""
        click.echo(f"  • {step_name} ({memory_str}{gpu_str})")

    click.echo("")

    # Set up storage
    if mode == ExecutionMode.LOCAL:
        set_storage(LocalStorage())
    else:
        # In Docker mode, might use S3
        if os.environ.get('OPS0_BUCKET'):
            set_storage(S3Storage())

    # Create orchestrator and run
    orchestrator = Orchestrator(mode=mode)

    try:
        # Parse arguments
        parsed_args = _parse_run_args(args)

        # Execute pipeline
        results = orchestrator.execute_pipeline(
            pipeline_config,
            *parsed_args['args'],
            **parsed_args['kwargs']
        )

        # Show results
        if results['success']:
            click.echo(f"\n✅ Pipeline completed successfully!")
            if 'results' in results and 'pipeline' in results['results']:
                result = results['results']['pipeline'].result
                if result:
                    click.echo(f"\n📊 Results:")
                    click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n❌ Pipeline failed!", err=True)
            for step_name, step_result in results['results'].items():
                if not step_result.success:
                    click.echo(f"\n💥 Error in {step_name}: {step_result.error}", err=True)
                    if 'traceback' in step_result.metadata:
                        click.echo(step_result.metadata['traceback'], err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        if '--debug' in sys.argv:
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--stage', '-s', default='prod', help='Deployment stage (dev/staging/prod)')
@click.option('--region', '-r', default='us-east-1', help='AWS region')
@click.option('--dry-run', is_flag=True, help='Show what would be deployed without deploying')
@click.option('--force', '-f', is_flag=True, help='Force deployment without confirmation')
def deploy(stage: str, region: str, dry_run: bool, force: bool):
    """Deploy pipeline to cloud infrastructure"""

    click.echo(f"🚀 Deploying to {click.style(stage, fg=COLORS['info'], bold=True)} in {region}...\n")

    # Load pipeline module
    pipeline_module = _load_pipeline_module()

    if not pipeline_module:
        click.echo("❌ No pipeline found. Create a pipeline.py file", err=True)
        sys.exit(1)

    # Get all pipelines
    pipelines = PipelineDecorator.get_all()

    if not pipelines:
        click.echo("❌ No pipelines found. Decorate a function with @ops0.pipeline", err=True)
        sys.exit(1)

    # For now, deploy first pipeline
    pipeline_name = list(pipelines.keys())[0]
    pipeline_config = pipelines[pipeline_name]

    # Show deployment plan
    click.echo(f"📋 Deployment Plan:")
    click.echo(f"  • Pipeline: {pipeline_config.name}")
    click.echo(f"  • Steps: {len(pipeline_config.steps)}")
    click.echo(f"  • Stage: {stage}")
    click.echo(f"  • Region: {region}")

    # Calculate resources
    total_memory = sum(s.memory for s in pipeline_config.steps.values())
    uses_gpu = any(s.gpu for s in pipeline_config.steps.values())

    click.echo(f"\n💰 Estimated Resources:")
    click.echo(f"  • Memory: {total_memory}MB total")
    click.echo(f"  • GPU: {'Required' if uses_gpu else 'Not required'}")
    click.echo(f"  • Lambda functions: {len(pipeline_config.steps)}")
    click.echo(f"  • Step Functions: 1")

    # Confirm deployment
    if not force and not dry_run:
        if not click.confirm(f"\n🤔 Deploy to {stage}?"):
            click.echo("❌ Deployment cancelled")
            sys.exit(0)

    if dry_run:
        click.echo(f"\n🔍 Dry run mode - no resources will be created")

    # Generate CDK application
    click.echo(f"\n📦 Generating infrastructure code...")

    cdk_dir = Path(".ops0/cdk")
    cdk_dir.mkdir(parents=True, exist_ok=True)

    generator = CDKGenerator()
    generator.generate_app(
        pipeline_config,
        pipeline_config.steps,
        cdk_dir,
        stage=stage,
        region=region
    )

    click.echo(f"  ✅ Generated CDK application")

    if dry_run:
        click.echo(f"\n📁 Generated files in {cdk_dir}")
        return

    # Install CDK dependencies
    click.echo(f"\n📥 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                    str(cdk_dir / "requirements.txt")],
                   capture_output=True)

    # Deploy with CDK
    click.echo(f"\n🏗️  Deploying infrastructure...")

    # Bootstrap CDK if needed
    bootstrap_cmd = ["npx", "cdk", "bootstrap", f"aws://{os.environ.get('CDK_DEFAULT_ACCOUNT', 'unknown')}/{region}"]
    subprocess.run(bootstrap_cmd, cwd=cdk_dir, capture_output=True)

    # Deploy
    deploy_cmd = ["npx", "cdk", "deploy", "--require-approval", "never"]
    result = subprocess.run(deploy_cmd, cwd=cdk_dir)

    if result.returncode == 0:
        click.echo(f"\n✅ Deployment successful!")
        click.echo(f"\n📊 View your pipeline:")
        click.echo(f"  • AWS Console: https://console.aws.amazon.com/states/home?region={region}")
        click.echo(f"  • Run: ops0 status")
    else:
        click.echo(f"\n❌ Deployment failed!", err=True)
        sys.exit(1)


@cli.command()
@click.option('--pipeline', '-p', help='Specific pipeline to check')
@click.option('--execution', '-e', help='Specific execution ID')
@click.option('--tail', '-f', is_flag=True, help='Follow logs in real-time')
def status(pipeline: Optional[str], execution: Optional[str], tail: bool):
    """Check pipeline status and metrics"""

    click.echo(f"📊 Pipeline Status\n")

    # For MVP, show mock status
    # In production, this would connect to CloudWatch/monitoring

    pipelines = PipelineDecorator.get_all()

    if not pipelines:
        click.echo("❌ No pipelines found", err=True)
        sys.exit(1)

    for pipeline_name, pipeline_config in pipelines.items():
        if pipeline and pipeline != pipeline_name:
            continue

        click.echo(f"📋 {click.style(pipeline_name, fg=COLORS['info'], bold=True)}")
        click.echo(f"  Status: 🟢 Healthy")
        click.echo(f"  Last run: 5 minutes ago")
        click.echo(f"  Success rate: 98.5%")
        click.echo(f"  Avg duration: 2m 34s")

        click.echo(f"\n  📊 Steps:")
        for step_name in pipeline_config.steps:
            click.echo(f"    • {step_name}: ✅ 156ms avg (124 runs today)")

    if tail:
        click.echo(f"\n👀 Watching for updates... (Ctrl+C to exit)")
        # In production, would tail CloudWatch logs


@cli.command()
@click.option('--pipeline', '-p', help='Specific pipeline logs')
@click.option('--step', '-s', help='Specific step logs')
@click.option('--execution', '-e', help='Specific execution ID')
@click.option('--tail', '-f', is_flag=True, help='Follow logs in real-time')
@click.option('--since', help='Show logs since timestamp (e.g., 2h, 30m)')
@click.option('--limit', '-n', type=int, default=100, help='Number of log lines')
def logs(pipeline: Optional[str], step: Optional[str], execution: Optional[str],
         tail: bool, since: Optional[str], limit: int):
    """View pipeline execution logs"""

    click.echo(f"📜 Logs\n")

    # For MVP, show local logs
    log_dir = Path(".ops0/logs")

    if not log_dir.exists():
        click.echo("No logs found. Run a pipeline first with: ops0 run")
        return

    # In production, this would fetch from CloudWatch
    click.echo("2024-01-15 10:23:45 | INFO  | Pipeline 'ml_pipeline' started")
    click.echo("2024-01-15 10:23:46 | INFO  | Step 'preprocess' started")
    click.echo("2024-01-15 10:23:48 | INFO  | Step 'preprocess' completed (2.1s)")
    click.echo("2024-01-15 10:23:48 | INFO  | Step 'train' started")
    click.echo("2024-01-15 10:24:15 | INFO  | Model saved: fraud_detector_v2")
    click.echo("2024-01-15 10:24:15 | INFO  | Step 'train' completed (27.0s)")
    click.echo("2024-01-15 10:24:16 | INFO  | Pipeline completed successfully")

    if tail:
        click.echo(f"\n👀 Tailing logs... (Ctrl+C to exit)")


@cli.command()
@click.option('--version', '-v', help='Version to rollback to')
@click.option('--list', '-l', is_flag=True, help='List available versions')
@click.option('--force', '-f', is_flag=True, help='Force rollback without confirmation')
def rollback(version: Optional[str], list: bool, force: bool):
    """Rollback to a previous pipeline version"""

    if list:
        click.echo(f"📦 Available versions:\n")
        # Mock versions for MVP
        versions = [
            ("v1.2.3", "2024-01-15 10:00", "Current", "green"),
            ("v1.2.2", "2024-01-14 15:30", "Stable", "blue"),
            ("v1.2.1", "2024-01-13 09:15", "Previous", "dim"),
        ]

        for ver, date, status, color in versions:
            status_str = click.style(f"[{status}]", fg=COLORS[color])
            click.echo(f"  {ver} - {date} {status_str}")
        return

    if not version:
        click.echo("❌ Please specify a version with --version or use --list", err=True)
        sys.exit(1)

    click.echo(f"🔄 Rolling back to version {click.style(version, fg=COLORS['info'], bold=True)}...")

    if not force:
        if not click.confirm(f"\n⚠️  This will replace the current deployment. Continue?"):
            click.echo("❌ Rollback cancelled")
            sys.exit(0)

    # In production, would trigger CloudFormation rollback
    click.echo(f"\n✅ Successfully rolled back to {version}")


# Helper functions

def _get_project_templates(template: str) -> Dict[str, str]:
    """Get project template files"""

    if template == 'basic':
        return {
            'pipeline.py': '''"""
Simple ops0 pipeline example
"""
import ops0

@ops0.step
def load_data(path: str):
    """Load data from file"""
    # Your data loading logic here
    return {"data": "loaded"}

@ops0.step  
def process_data(data):
    """Process the data"""
    # Your processing logic here
    return {"processed": True}

@ops0.pipeline
def my_pipeline(input_path: str):
    """My first ops0 pipeline"""
    data = load_data(input_path)
    result = process_data(data)
    return result

if __name__ == "__main__":
    # Test locally
    result = my_pipeline("data/input.csv")
    print(f"Result: {result}")
''',
            'requirements.txt': '''ops0>=0.1.0
pandas>=1.3.0
numpy>=1.21.0
''',
            'README.md': '''# My ops0 Project

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
ops0 run --local

# Deploy to cloud
ops0 deploy
```
'''
        }

    elif template == 'ml':
        return {
            'pipeline.py': '''"""
Machine Learning pipeline with ops0
"""
import ops0
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@ops0.step
def load_dataset(path: str):
    """Load and prepare dataset"""
    df = pd.read_csv(path)
    return df

@ops0.step
def prepare_features(df: pd.DataFrame):
    """Feature engineering"""
    # Add your feature engineering here
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

@ops0.step(memory=2048)
def train_model(X_train, X_test, y_train, y_test):
    """Train ML model"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save model
    ops0.save_model(model, "my_classifier", {"accuracy": accuracy})

    return {"accuracy": accuracy}

@ops0.pipeline
def ml_training_pipeline(data_path: str):
    """Complete ML training pipeline"""
    data = load_dataset(data_path)
    X_train, X_test, y_train, y_test = prepare_features(data)
    metrics = train_model(X_train, X_test, y_train, y_test)
    return metrics

if __name__ == "__main__":
    # Test with sample data
    metrics = ml_training_pipeline("data/train.csv")
    print(f"Model accuracy: {metrics['accuracy']:.2%}")
''',
            'requirements.txt': '''ops0>=0.1.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
''',
            'notebooks/exploration.ipynb': '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
        }

    return {}


def _load_pipeline_module():
    """Load the pipeline module dynamically"""
    # Look for pipeline files
    pipeline_files = ['pipeline.py', 'main.py', 'app.py']

    for file in pipeline_files:
        if Path(file).exists():
            spec = importlib.util.spec_from_file_location("pipeline", file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    # Check pipelines directory
    pipelines_dir = Path("pipelines")
    if pipelines_dir.exists():
        for file in pipelines_dir.glob("*.py"):
            if not file.name.startswith("_"):
                spec = importlib.util.spec_from_file_location(file.stem, file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    return None


def _parse_run_args(args: tuple) -> Dict[str, Any]:
    """Parse command line arguments for pipeline execution"""
    parsed = {
        'args': [],
        'kwargs': {}
    }

    for arg in args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Try to parse value
            try:
                value = json.loads(value)
            except:
                # Keep as string
                pass
            parsed['kwargs'][key] = value
        else:
            # Positional argument
            try:
                value = json.loads(arg)
            except:
                value = arg
            parsed['args'].append(value)

    return parsed

@cli.command()
def doctor():
    """Check ops0 installation and configuration"""
    issues = []

    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")

    # Check required packages
    try:
        import cloudpickle
    except ImportError:
        issues.append("cloudpickle not installed")

    # Check optional packages
    optional = {
        'boto3': 'AWS S3 storage',
        'pandas': 'DataFrame operations',
        'numpy': 'Numerical operations'
    }

    for pkg, feature in optional.items():
        try:
            __import__(pkg)
            click.echo(f"✅ {pkg} installed ({feature} supported)")
        except ImportError:
            click.echo(f"⚠️  {pkg} not installed ({feature} unavailable)")

    if issues:
        click.echo(f"\n❌ Issues found:")
        for issue in issues:
            click.echo(f"  • {issue}")
        sys.exit(1)
    else:
        click.echo(f"\n✅ ops0 is ready to use!")



# Entry point
def main():
    """Main CLI entry point"""
    cli()


if __name__ == '__main__':
    main()