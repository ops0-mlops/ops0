import click
import sys
import os
from pathlib import Path
from core.graph import PipelineGraph
from core.executor import run, deploy


@click.group()
@click.version_option()
def cli():
    """ops0 - Python-Native ML Pipeline Orchestration"""
    pass


@cli.command()
@click.option('--local', is_flag=True, help='Run pipeline locally')
@click.option('--pipeline', help='Pipeline file to run')
def run_cmd(local, pipeline):
    """Run a pipeline"""
    mode = "local" if local else "distributed"

    if pipeline:
        # Load and execute pipeline from file
        # This would import the Python file and execute it
        click.echo(f"Running pipeline from {pipeline} in {mode} mode")

    try:
        results = run(mode=mode)
        click.echo(f"✅ Pipeline completed successfully!")
        click.echo(f"📊 Results: {len(results)} steps executed")
    except Exception as e:
        click.echo(f"❌ Pipeline failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--env', default='production', help='Deployment environment')
@click.option('--name', help='Pipeline name')
def deploy_cmd(env, name):
    """Deploy pipeline to production"""
    try:
        result = deploy(name=name, env=env)
        click.echo(f"✅ Deployment successful!")
        click.echo(f"🔗 URL: {result['url']}")
    except Exception as e:
        click.echo(f"❌ Deployment failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def init():
    """Initialize a new ops0 project"""
    click.echo("🚀 Initializing new ops0 project...")

    # Create project structure
    dirs = [".ops0", ".ops0/storage", "pipelines"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

    # Create example pipeline
    example_pipeline = '''
import ops0
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@ops0.step
def load_data():
    """Load training data"""
    # Replace with your data loading logic
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [0, 1, 0, 1, 0]
    })
    ops0.storage.save("raw_data", data)
    return data

@ops0.step
def preprocess():
    """Preprocess the data"""
    data = ops0.storage.load("raw_data")

    # Simple preprocessing
    processed = data.copy()
    processed['feature1_scaled'] = processed['feature1'] / processed['feature1'].max()

    ops0.storage.save("processed_data", processed)
    return processed

@ops0.step
def train_model():
    """Train a simple model"""
    data = ops0.storage.load("processed_data")

    X = data[['feature1_scaled', 'feature2']]
    y = data['target']

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    ops0.storage.save("trained_model", model)
    return {"accuracy": model.score(X, y)}

if __name__ == "__main__":
    with ops0.pipeline("example-ml-pipeline"):
        # Define the pipeline steps
        load_data()
        preprocess() 
        train_model()

        # Run locally
        results = ops0.run(mode="local")
        print(f"Pipeline results: {results}")
'''

    with open("pipelines/example.py", "w") as f:
        f.write(example_pipeline.strip())

    click.echo("📁 Created project structure:")
    click.echo("   ├── .ops0/           # ops0 metadata")
    click.echo("   ├── pipelines/       # Your pipeline code")
    click.echo("   └── pipelines/example.py  # Example pipeline")
    click.echo("\n🎯 Next steps:")
    click.echo("   1. Edit pipelines/example.py")
    click.echo("   2. Run: python pipelines/example.py")
    click.echo("   3. Deploy: ops0 deploy")


@cli.command()
def status():
    """Show pipeline status"""
    current = PipelineGraph.get_current()
    if current:
        click.echo(f"📊 Current Pipeline: {current.name}")
        click.echo(f"📋 Steps: {len(current.steps)}")
        click.echo("\n" + current.visualize())
    else:
        click.echo("No active pipeline found")


@cli.command()
@click.option('--push/--no-push', default=False, help='Push containers to registry')
@click.option('--step', help='Containerize specific step only')
def containerize(push, step):
    """Containerize pipeline steps automatically"""
    current = PipelineGraph.get_current()
    if not current:
        click.echo("❌ No active pipeline found", err=True)
        return

    # Import here to avoid circular dependency
    from runtime.containers import container_orchestrator

    if step:
        if step not in current.steps:
            click.echo(f"❌ Step '{step}' not found in pipeline", err=True)
            return
        click.echo(f"🐳 Containerizing step: {step}")
        # Containerize single step logic would go here
    else:
        click.echo(f"🐳 Containerizing entire pipeline: {current.name}")

        # Set environment variable for building
        if push:
            os.environ["OPS0_BUILD_CONTAINERS"] = "true"

        specs = container_orchestrator.containerize_pipeline(current)

        click.echo(f"\n📋 Containerization Summary:")
        for step_name, spec in specs.items():
            click.echo(f"  ├─ {step_name}: {spec.image_tag}")
            click.echo(f"  │  ├─ Memory: {spec.memory_limit}")
            click.echo(f"  │  ├─ GPU: {'Yes' if spec.needs_gpu else 'No'}")
            click.echo(f"  │  └─ Requirements: {len(spec.requirements)} packages")


@cli.command()
def build():
    """Build containers for current pipeline"""
    current = PipelineGraph.get_current()
    if not current:
        click.echo("❌ No active pipeline found", err=True)
        return

    # Set build flag
    os.environ["OPS0_BUILD_CONTAINERS"] = "true"

    from runtime.containers import container_orchestrator

    click.echo(f"🔨 Building containers for pipeline: {current.name}")
    specs = container_orchestrator.containerize_pipeline(current)

    click.echo(f"✅ Built {len(specs)} containers successfully!")


@cli.command()
def manifest():
    """Generate deployment manifest"""
    current = PipelineGraph.get_current()
    if not current:
        click.echo("❌ No active pipeline found", err=True)
        return

    from runtime.containers import container_orchestrator
    import json

    # Generate manifest
    manifest = container_orchestrator.get_container_manifest()

    # Pretty print manifest
    click.echo("📋 Deployment Manifest:")
    click.echo("=" * 30)
    click.echo(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    cli()
