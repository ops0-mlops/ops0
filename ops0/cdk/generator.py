"""
AWS CDK generator for ops0 pipelines.
Automatically generates infrastructure as code from Python functions.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import textwrap

from ops0.decorators import PipelineConfig, StepConfig
from ops0.parser import build_dag


@dataclass
class CDKConfig:
    """Configuration for CDK generation"""
    app_name: str
    stage: str
    region: str
    account: Optional[str] = None
    vpc_id: Optional[str] = None


class CDKGenerator:
    """Generate AWS CDK applications from ops0 pipelines"""

    def __init__(self):
        self.templates_dir = Path(__file__).parent / "templates"

    def generate_app(
            self,
            pipeline_config: PipelineConfig,
            steps: Dict[str, StepConfig],
            output_dir: Path,
            stage: str = "prod",
            region: str = "us-east-1"
    ):
        """Generate complete CDK application"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate CDK app structure
        self._generate_app_py(output_dir, pipeline_config.name, stage, region)
        self._generate_cdk_json(output_dir)
        self._generate_requirements(output_dir)

        # Generate stack
        stack_dir = output_dir / "stacks"
        stack_dir.mkdir(exist_ok=True)

        self._generate_pipeline_stack(
            stack_dir,
            pipeline_config,
            steps,
            stage
        )

        # Generate Lambda functions
        lambda_dir = output_dir / "lambda"
        lambda_dir.mkdir(exist_ok=True)

        for step_name, step_config in steps.items():
            self._generate_lambda_function(lambda_dir, step_config)

    def _generate_app_py(self, output_dir: Path, app_name: str, stage: str, region: str):
        """Generate CDK app.py file"""
        content = f'''#!/usr/bin/env python3
import os
import aws_cdk as cdk
from stacks.pipeline_stack import PipelineStack

app = cdk.App()

PipelineStack(
    app,
    f"{app_name}-{{stage}}-stack",
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region="{region}"
    ),
    stage="{stage}"
)

app.synth()
'''
        (output_dir / "app.py").write_text(content)

    def _generate_cdk_json(self, output_dir: Path):
        """Generate cdk.json configuration"""
        config = {
            "app": "python3 app.py",
            "context": {
                "@aws-cdk/core:enableStackNameDuplicates": True,
                "@aws-cdk/core:stackRelativeExports": True,
                "@aws-cdk/aws-lambda:recognizeVersionProps": True
            }
        }

        with open(output_dir / "cdk.json", "w") as f:
            json.dump(config, f, indent=2)

    def _generate_requirements(self, output_dir: Path):
        """Generate requirements.txt for CDK"""
        requirements = """aws-cdk-lib>=2.100.0
constructs>=10.0.0
"""
        (output_dir / "requirements.txt").write_text(requirements)

    def _generate_pipeline_stack(
            self,
            stack_dir: Path,
            pipeline_config: PipelineConfig,
            steps: Dict[str, StepConfig],
            stage: str
    ):
        """Generate the main CDK stack with Step Functions"""

        # Build DAG to understand dependencies
        dag = build_dag(pipeline_config.func, steps)

        # Generate CDK stack code
        content = f'''from aws_cdk import (
    Stack,
    aws_lambda as lambda_,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_s3 as s3,
    aws_logs as logs,
    Duration,
    RemovalPolicy
)
from constructs import Construct


class PipelineStack(Stack):
    """CDK stack for {pipeline_config.name} pipeline"""

    def __init__(self, scope: Construct, construct_id: str, stage: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # S3 bucket for pipeline storage
        self.bucket = s3.Bucket(
            self, "PipelineBucket",
            bucket_name=f"ops0-{pipeline_config.name}-{{stage}}",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        # Create Lambda functions for each step
        self.lambdas = {{}}

'''

        # Generate Lambda functions
        for step_name, step_config in steps.items():
            memory = step_config.memory
            timeout = step_config.timeout

            content += f'''        self.lambdas["{step_name}"] = lambda_.Function(
            self, "{step_name.title()}Function",
            function_name=f"ops0-{{stage}}-{step_name}",
            runtime=lambda_.Runtime.PYTHON_3_9,
            code=lambda_.Code.from_asset("lambda/{step_name}"),
            handler="handler.main",
            memory_size={memory},
            timeout=Duration.seconds({timeout}),
            environment={{
                "OPS0_BUCKET": self.bucket.bucket_name,
                "OPS0_STAGE": stage
            }}
        )

        # Grant bucket permissions
        self.bucket.grant_read_write(self.lambdas["{step_name}"])

'''

        # Generate Step Functions state machine
        content += '''        # Create Step Functions tasks
        '''

        # Create tasks for each step
        for step_name in steps:
            content += f'''
        {step_name}_task = tasks.LambdaInvoke(
            self, "{step_name.title()}Task",
            lambda_function=self.lambdas["{step_name}"],
            output_path="$.Payload"
        )
'''

        # Build state machine definition based on DAG
        content += f'''
        # Build state machine
        definition = {self._build_sfn_definition(dag, pipeline_config.name)}

        # Create state machine
        self.state_machine = sfn.StateMachine(
            self, "PipelineStateMachine",
           state_machine_name=f"ops0-{stage}-{pipeline_config.name}",
            definition=definition,
            logs={{
                "destination": logs.LogGroup(
                    self, "StateMachineLogGroup",
                    removal_policy=RemovalPolicy.DESTROY
                ),
                "level": sfn.LogLevel.ALL
            }}
        )
'''

        (stack_dir / "pipeline_stack.py").write_text(content)

    def _build_sfn_definition(self, dag: Dict[str, List[str]], pipeline_name: str) -> str:
        """Build Step Functions definition from DAG"""
        # For MVP, create a simple sequential chain
        # Full implementation would handle parallel execution

        # Find steps with no dependencies (entry points)
        all_steps = set(dag.keys())
        all_deps = set()
        for deps in dag.values():
            all_deps.update(deps)
        entry_points = all_steps - all_deps

        if not entry_points:
            # If no clear entry point, use first step
            entry_points = {list(dag.keys())[0]}

        # For now, build sequential chain
        step_order = list(dag.keys())

        definition = f"{step_order[0]}_task"
        for i in range(1, len(step_order)):
            definition = f"{definition}.next({step_order[i]}_task)"

        return definition

    def _generate_lambda_function(self, lambda_dir: Path, step_config: StepConfig):
        """Generate Lambda function code for a step"""
        step_dir = lambda_dir / step_config.name
        step_dir.mkdir(exist_ok=True)

        # Generate handler
        handler_content = f'''"""
Lambda handler for {step_config.name} step
Auto-generated by ops0
"""
import os
import json
import boto3
import cloudpickle

# Initialize S3 client
s3 = boto3.client('s3')
bucket = os.environ['OPS0_BUCKET']


def main(event, context):
    """Lambda handler"""
    try:
        # Load inputs from event or S3
        if 'input_key' in event:
            # Load from S3
            response = s3.get_object(Bucket=bucket, Key=event['input_key'])
            inputs = cloudpickle.loads(response['Body'].read())
        else:
            # Direct inputs
            inputs = event.get('inputs', {{}})

        # Execute step function
        # In production, this would deserialize and run the actual function
        result = {{"status": "success", "step": "{step_config.name}"}}

        # Save results to S3
        output_key = f"outputs/{step_config.name}/{{context.request_id}}.pkl"
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=cloudpickle.dumps(result)
        )

        return {{
            "statusCode": 200,
            "output_key": output_key,
            "result": result
        }}

    except Exception as e:
        return {{
            "statusCode": 500,
            "error": str(e)
        }}
'''

        (step_dir / "handler.py").write_text(handler_content)

        # Generate requirements
        requirements = ["cloudpickle>=2.0.0", "boto3>=1.20.0"]
        requirements.extend(step_config.requirements)

        (step_dir / "requirements.txt").write_text("\n".join(requirements))