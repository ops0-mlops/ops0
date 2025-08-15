"""
ops0 CDK module - Infrastructure as Code generation for ML pipelines.

This module automatically generates AWS CDK applications from ops0 pipelines,
enabling zero-configuration deployment to AWS infrastructure.

Usage:
    from ops0.cdk import CDKGenerator

    generator = CDKGenerator()
    generator.generate_app(pipeline_config, steps, output_dir)
"""

from ops0.cdk.generator import (
    CDKGenerator,
    CDKConfig
)

# Version info
__version__ = "0.1.0"

# Public API
__all__ = [
    "CDKGenerator",
    "CDKConfig",
    "generate_infrastructure",
    "validate_cdk_config"
]


def generate_infrastructure(
    pipeline_config,
    output_dir="./cdk.out",
    stage="prod",
    region="us-east-1",
    account=None
):
    """
    High-level function to generate CDK infrastructure from a pipeline.

    Args:
        pipeline_config: Pipeline configuration object
        output_dir: Directory to output CDK app (default: ./cdk.out)
        stage: Deployment stage (dev/staging/prod)
        region: AWS region
        account: AWS account ID (optional)

    Returns:
        Path to generated CDK application

    Example:
        from ops0.cdk import generate_infrastructure

        path = generate_infrastructure(
            my_pipeline_config,
            stage="dev",
            region="eu-west-1"
        )
    """
    from pathlib import Path
    from ops0.decorators import StepDecorator

    # Get all steps associated with the pipeline
    steps = {}
    for step_name, step_config in pipeline_config.steps.items():
        steps[step_name] = step_config

    # Create generator and generate app
    generator = CDKGenerator()
    generator.generate_app(
        pipeline_config=pipeline_config,
        steps=steps,
        output_dir=Path(output_dir),
        stage=stage,
        region=region
    )

    return Path(output_dir)


def validate_cdk_config(config: CDKConfig) -> bool:
    """
    Validate CDK configuration before generation.

    Args:
        config: CDKConfig object to validate

    Returns:
        True if valid, raises ValueError if invalid

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate stage
    valid_stages = ["dev", "staging", "prod", "test"]
    if config.stage not in valid_stages:
        raise ValueError(f"Invalid stage '{config.stage}'. Must be one of: {valid_stages}")

    # Validate region
    aws_regions = [
        "us-east-1", "us-east-2", "us-west-1", "us-west-2",
        "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1",
        "ap-southeast-1", "ap-southeast-2", "ap-northeast-1",
        "sa-east-1", "ca-central-1"
    ]
    if config.region not in aws_regions:
        raise ValueError(f"Invalid AWS region '{config.region}'")

    # Validate app name
    if not config.app_name or len(config.app_name) < 3:
        raise ValueError("App name must be at least 3 characters long")

    # Validate account if provided
    if config.account and not config.account.isdigit():
        raise ValueError("AWS account must be numeric")

    return True


# CDK Template Registry - For future extensibility
_CDK_TEMPLATES = {
    "lambda_step_functions": "Default serverless template using Lambda + Step Functions",
    "ecs_fargate": "Container-based template using ECS Fargate (future)",
    "sagemaker": "SageMaker-based template for ML workloads (future)",
    "batch": "AWS Batch template for large-scale processing (future)"
}


def list_cdk_templates():
    """
    List available CDK templates for pipeline deployment.

    Returns:
        Dict of template_name -> description
    """
    return _CDK_TEMPLATES.copy()


def get_template_info(template_name: str) -> dict:
    """
    Get detailed information about a CDK template.

    Args:
        template_name: Name of the template

    Returns:
        Dict with template information

    Raises:
        KeyError: If template doesn't exist
    """
    if template_name not in _CDK_TEMPLATES:
        raise KeyError(f"Template '{template_name}' not found. Available: {list(_CDK_TEMPLATES.keys())}")

    # For now, return basic info. In future, this would include
    # detailed template metadata, requirements, cost estimates, etc.
    return {
        "name": template_name,
        "description": _CDK_TEMPLATES[template_name],
        "supported": template_name == "lambda_step_functions",
        "requirements": {
            "aws_services": ["Lambda", "Step Functions", "S3", "CloudWatch"],
            "permissions": ["iam:CreateRole", "lambda:CreateFunction", "states:CreateStateMachine"]
        }
    }


# Infrastructure cost estimation (simplified for MVP)
def estimate_cost(pipeline_config, executions_per_month=1000) -> dict:
    """
    Estimate monthly AWS costs for a pipeline.

    Args:
        pipeline_config: Pipeline configuration
        executions_per_month: Expected monthly executions

    Returns:
        Dict with cost breakdown

    Note:
        This is a simplified estimation. Actual costs may vary.
    """
    # Lambda pricing (simplified)
    LAMBDA_PRICE_PER_GB_SECOND = 0.0000166667
    LAMBDA_PRICE_PER_REQUEST = 0.0000002

    # Step Functions pricing
    STEP_FUNCTIONS_PRICE_PER_1000_TRANSITIONS = 0.025

    total_cost = 0.0
    breakdown = {
        "lambda_compute": 0.0,
        "lambda_requests": 0.0,
        "step_functions": 0.0,
        "s3_storage": 5.0,  # Rough estimate
        "cloudwatch": 3.0   # Rough estimate
    }

    # Calculate Lambda costs
    for step_name, step_config in pipeline_config.steps.items():
        # Assume average execution time of 10 seconds per step
        avg_execution_seconds = 10
        gb_seconds = (step_config.memory / 1024) * avg_execution_seconds * executions_per_month

        breakdown["lambda_compute"] += gb_seconds * LAMBDA_PRICE_PER_GB_SECOND
        breakdown["lambda_requests"] += executions_per_month * LAMBDA_PRICE_PER_REQUEST

    # Calculate Step Functions costs
    # Assume 2 state transitions per step (start + end)
    total_transitions = len(pipeline_config.steps) * 2 * executions_per_month
    breakdown["step_functions"] = (total_transitions / 1000) * STEP_FUNCTIONS_PRICE_PER_1000_TRANSITIONS

    # Total
    total_cost = sum(breakdown.values())

    return {
        "total_monthly_cost_usd": round(total_cost, 2),
        "breakdown": {k: round(v, 2) for k, v in breakdown.items()},
        "assumptions": {
            "executions_per_month": executions_per_month,
            "avg_execution_seconds_per_step": 10,
            "includes_free_tier": False
        }
    }


# Utility function for cleaning up CDK output
def cleanup_cdk_output(output_dir="./cdk.out"):
    """
    Clean up generated CDK output directory.

    Args:
        output_dir: Path to CDK output directory
    """
    import shutil
    from pathlib import Path

    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"✅ Cleaned up CDK output at {output_dir}")
    else:
        print(f"📁 No CDK output found at {output_dir}")