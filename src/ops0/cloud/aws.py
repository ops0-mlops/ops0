"""
ops0 AWS Provider

Zero-configuration AWS deployment for ML pipelines.
Automatically handles ECS, Lambda, S3, SQS, and more.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .base import (
    CloudProvider, CloudResource, DeploymentSpec,
    ResourceType, ResourceStatus, ResourceMetrics
)

logger = logging.getLogger(__name__)


class AWSProvider(CloudProvider):
    """AWS cloud provider implementation"""

    def _setup(self):
        """Initialize AWS provider"""
        # Auto-detect configuration
        self.region = (
                self.config.get('region') or
                os.environ.get('AWS_DEFAULT_REGION') or
                os.environ.get('AWS_REGION') or
                'us-east-1'
        )

        self.account_id = self.config.get('account_id', '123456789012')

        # Initialize boto3 clients lazily
        self._clients = {}

        # ECS cluster for containers
        self.ecs_cluster = f"ops0-cluster-{self.region}"

        # Lambda layer for common dependencies
        self.lambda_layer_arn = None

        logger.info(f"AWS provider initialized for region: {self.region}")

    @property
    def name(self) -> str:
        return "aws"

    @property
    def regions(self) -> List[str]:
        return [
            'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
            'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1',
            'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1',
            'ap-south-1', 'sa-east-1'
        ]

    def _get_client(self, service: str):
        """Get or create boto3 client"""
        if service not in self._clients:
            try:
                import boto3
                self._clients[service] = boto3.client(service, region_name=self.region)
            except ImportError:
                logger.warning("boto3 not installed - using mock mode")
                from unittest.mock import MagicMock
                self._clients[service] = MagicMock()

        return self._clients[service]

    def deploy_container(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy container to ECS Fargate"""
        ecs = self._get_client('ecs')

        # Create task definition
        task_family = f"ops0-{spec.step_name}"
        container_name = spec.step_name

        task_definition = {
            'family': task_family,
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': str(int(spec.cpu * 1024)),  # Fargate CPU units
            'memory': str(spec.memory),
            'containerDefinitions': [{
                'name': container_name,
                'image': spec.image,
                'command': spec.command,
                'environment': [
                    {'name': k, 'value': v}
                    for k, v in spec.environment.items()
                ],
                'logConfiguration': {
                    'logDriver': 'awslogs',
                    'options': {
                        'awslogs-group': f'/ecs/ops0/{spec.step_name}',
                        'awslogs-region': self.region,
                        'awslogs-stream-prefix': 'ecs'
                    }
                },
                'essential': True
            }]
        }

        # Add GPU if required
        if spec.gpu > 0:
            task_definition['containerDefinitions'][0]['resourceRequirements'] = [{
                'type': 'GPU',
                'value': str(spec.gpu)
            }]

        # Register task definition
        response = ecs.register_task_definition(**task_definition)
        task_def_arn = response['taskDefinition']['taskDefinitionArn']

        # Create or update service
        service_name = f"ops0-{spec.step_name}-service"

        try:
            # Check if service exists
            ecs.describe_services(
                cluster=self.ecs_cluster,
                services=[service_name]
            )

            # Update existing service
            response = ecs.update_service(
                cluster=self.ecs_cluster,
                service=service_name,
                taskDefinition=task_def_arn,
                desiredCount=spec.min_instances,
                deploymentConfiguration={
                    'maximumPercent': 200,
                    'minimumHealthyPercent': 100
                }
            )

        except ecs.exceptions.ServiceNotFoundException:
            # Create new service
            subnets = self._get_subnets()
            security_group = self._get_security_group()

            response = ecs.create_service(
                cluster=self.ecs_cluster,
                serviceName=service_name,
                taskDefinition=task_def_arn,
                desiredCount=spec.min_instances,
                launchType='FARGATE' if not spec.spot_instances else 'FARGATE_SPOT',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': subnets,
                        'securityGroups': [security_group],
                        'assignPublicIp': 'ENABLED'
                    }
                },
                deploymentConfiguration={
                    'maximumPercent': 200,
                    'minimumHealthyPercent': 100
                }
            )

        # Create auto-scaling
        self._setup_auto_scaling(service_name, spec)

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.CONTAINER,
            resource_id=service_name,
            resource_name=service_name,
            region=self.region,
            status=ResourceStatus.PROVISIONING,
            metadata={
                'cluster': self.ecs_cluster,
                'task_definition': task_def_arn,
                'service_arn': response['service']['serviceArn'],
                'launch_type': 'FARGATE_SPOT' if spec.spot_instances else 'FARGATE'
            },
            tags={
                'step': spec.step_name,
                'environment': self.config.get('environment', 'production')
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Deployed container to ECS: {service_name}")

        return resource

    def deploy_function(self, spec: DeploymentSpec) -> CloudResource:
        """Deploy serverless function to Lambda"""
        lambda_client = self._get_client('lambda')

        function_name = f"ops0-{spec.step_name}"

        # Package code (in production, this would zip the actual code)
        code_package = self._package_lambda_code(spec)

        # Lambda configuration
        lambda_config = {
            'FunctionName': function_name,
            'Runtime': 'python3.9',
            'Role': self._get_lambda_role(),
            'Handler': 'handler.main',
            'Code': code_package,
            'Environment': {
                'Variables': spec.environment
            },
            'MemorySize': spec.memory,
            'Timeout': spec.timeout_seconds,
            'TracingConfig': {
                'Mode': 'Active'  # X-Ray tracing
            }
        }

        # Add layers for ML frameworks
        if any(lib in spec.image for lib in ['numpy', 'pandas', 'scikit-learn', 'tensorflow', 'pytorch']):
            lambda_config['Layers'] = [self._get_ml_layer_arn()]

        try:
            # Create function
            response = lambda_client.create_function(**lambda_config)
            function_arn = response['FunctionArn']

        except lambda_client.exceptions.ResourceConflictException:
            # Update existing function
            response = lambda_client.update_function_configuration(
                FunctionName=function_name,
                Environment={'Variables': spec.environment},
                MemorySize=spec.memory,
                Timeout=spec.timeout_seconds
            )

            # Update code
            lambda_client.update_function_code(
                FunctionName=function_name,
                **code_package
            )

            function_arn = response['FunctionArn']

        # Set up concurrency and scaling
        lambda_client.put_function_concurrency(
            FunctionName=function_name,
            ReservedConcurrentExecutions=spec.max_instances * 10
        )

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.FUNCTION,
            resource_id=function_name,
            resource_name=function_name,
            region=self.region,
            status=ResourceStatus.RUNNING,
            metadata={
                'function_arn': function_arn,
                'runtime': 'python3.9',
                'memory_mb': spec.memory,
                'timeout_seconds': spec.timeout_seconds
            },
            tags={
                'step': spec.step_name,
                'type': 'serverless'
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Deployed function to Lambda: {function_name}")

        return resource

    def create_storage(self, name: str, region: str, **kwargs) -> CloudResource:
        """Create S3 bucket"""
        s3 = self._get_client('s3')

        bucket_name = f"ops0-{name}-{int(time.time())}"

        # Create bucket
        if region == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )

        # Enable versioning
        s3.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )

        # Set lifecycle policy for cost optimization
        lifecycle_config = {
            'Rules': [{
                'ID': 'ops0-lifecycle',
                'Status': 'Enabled',
                'Transitions': [
                    {
                        'Days': 30,
                        'StorageClass': 'INTELLIGENT_TIERING'
                    },
                    {
                        'Days': 90,
                        'StorageClass': 'GLACIER'
                    }
                ]
            }]
        }

        s3.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_config
        )

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.STORAGE,
            resource_id=bucket_name,
            resource_name=bucket_name,
            region=region,
            status=ResourceStatus.RUNNING,
            metadata={
                'endpoint': f"https://{bucket_name}.s3.{region}.amazonaws.com",
                'versioning': True,
                'lifecycle': 'enabled'
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created S3 bucket: {bucket_name}")

        return resource

    def create_queue(self, name: str, region: str, **kwargs) -> CloudResource:
        """Create SQS queue"""
        sqs = self._get_client('sqs')

        queue_name = f"ops0-{name}"

        # Create queue with optimized settings for ML workloads
        response = sqs.create_queue(
            QueueName=queue_name,
            Attributes={
                'MessageRetentionPeriod': str(kwargs.get('retention_days', 4) * 86400),
                'VisibilityTimeout': str(kwargs.get('visibility_timeout', 300)),
                'ReceiveMessageWaitTimeSeconds': '20',  # Long polling
                'MaximumMessageSize': '262144'  # 256KB
            }
        )

        queue_url = response['QueueUrl']

        # Get queue ARN
        attrs = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['QueueArn']
        )
        queue_arn = attrs['Attributes']['QueueArn']

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.QUEUE,
            resource_id=queue_name,
            resource_name=queue_name,
            region=region,
            status=ResourceStatus.RUNNING,
            metadata={
                'queue_url': queue_url,
                'queue_arn': queue_arn
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created SQS queue: {queue_name}")

        return resource

    def create_secret(self, name: str, value: str, region: str) -> CloudResource:
        """Store secret in AWS Secrets Manager"""
        secrets = self._get_client('secretsmanager')

        secret_name = f"ops0/{name}"

        try:
            response = secrets.create_secret(
                Name=secret_name,
                SecretString=value,
                Tags=[
                    {'Key': 'managed-by', 'Value': 'ops0'}
                ]
            )
            secret_arn = response['ARN']

        except secrets.exceptions.ResourceExistsException:
            # Update existing secret
            response = secrets.update_secret(
                SecretId=secret_name,
                SecretString=value
            )
            secret_arn = response['ARN']

        # Create CloudResource
        resource = CloudResource(
            provider=self.name,
            resource_type=ResourceType.SECRET,
            resource_id=secret_name,
            resource_name=secret_name,
            region=region,
            status=ResourceStatus.RUNNING,
            metadata={
                'secret_arn': secret_arn
            }
        )

        self.resources[resource.resource_id] = resource
        logger.info(f"Created secret: {secret_name}")

        return resource

    def get_resource_status(self, resource_id: str) -> ResourceStatus:
        """Get current status of a resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            return ResourceStatus.DELETED

        if resource.resource_type == ResourceType.CONTAINER:
            ecs = self._get_client('ecs')

            try:
                response = ecs.describe_services(
                    cluster=self.ecs_cluster,
                    services=[resource_id]
                )

                service = response['services'][0]

                if service['status'] == 'ACTIVE':
                    if service['runningCount'] == service['desiredCount']:
                        return ResourceStatus.RUNNING
                    else:
                        return ResourceStatus.UPDATING
                elif service['status'] == 'DRAINING':
                    return ResourceStatus.STOPPING
                else:
                    return ResourceStatus.STOPPED

            except Exception as e:
                logger.error(f"Error checking ECS service status: {e}")
                return ResourceStatus.FAILED

        elif resource.resource_type == ResourceType.FUNCTION:
            lambda_client = self._get_client('lambda')

            try:
                response = lambda_client.get_function(
                    FunctionName=resource_id
                )

                state = response['Configuration']['State']

                if state == 'Active':
                    return ResourceStatus.RUNNING
                elif state == 'Pending':
                    return ResourceStatus.PROVISIONING
                else:
                    return ResourceStatus.FAILED

            except Exception as e:
                logger.error(f"Error checking Lambda status: {e}")
                return ResourceStatus.FAILED

        # For other resources, assume running
        return ResourceStatus.RUNNING

    def get_resource_metrics(self, resource_id: str) -> ResourceMetrics:
        """Get CloudWatch metrics for a resource"""
        cloudwatch = self._get_client('cloudwatch')
        resource = self.resources.get(resource_id)

        if not resource:
            return ResourceMetrics()

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)

        metrics = ResourceMetrics()

        if resource.resource_type == ResourceType.CONTAINER:
            # Get ECS metrics
            namespace = 'AWS/ECS'
            dimensions = [
                {'Name': 'ServiceName', 'Value': resource_id},
                {'Name': 'ClusterName', 'Value': self.ecs_cluster}
            ]

            # CPU utilization
            cpu_response = cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName='CPUUtilization',
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )

            if cpu_response['Datapoints']:
                metrics.cpu_percent = cpu_response['Datapoints'][0]['Average']

            # Memory utilization
            mem_response = cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName='MemoryUtilization',
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )

            if mem_response['Datapoints']:
                metrics.memory_percent = mem_response['Datapoints'][0]['Average']

        elif resource.resource_type == ResourceType.FUNCTION:
            # Get Lambda metrics
            namespace = 'AWS/Lambda'
            dimensions = [
                {'Name': 'FunctionName', 'Value': resource_id}
            ]

            # Invocations
            inv_response = cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName='Invocations',
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Sum']
            )

            if inv_response['Datapoints']:
                metrics.request_count = int(inv_response['Datapoints'][0]['Sum'])

            # Duration
            dur_response = cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName='Duration',
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )

            if dur_response['Datapoints']:
                metrics.average_latency_ms = dur_response['Datapoints'][0]['Average']

        return metrics

    def update_resource(self, resource_id: str, spec: DeploymentSpec) -> CloudResource:
        """Update a deployed resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            raise ValueError(f"Resource not found: {resource_id}")

        if resource.resource_type == ResourceType.CONTAINER:
            # Update ECS service
            ecs = self._get_client('ecs')

            # Update task definition
            updated_resource = self.deploy_container(spec)

            # Force new deployment
            ecs.update_service(
                cluster=self.ecs_cluster,
                service=resource_id,
                forceNewDeployment=True
            )

            resource.status = ResourceStatus.UPDATING
            resource.updated_at = datetime.now()

        elif resource.resource_type == ResourceType.FUNCTION:
            # Update Lambda function
            lambda_client = self._get_client('lambda')

            lambda_client.update_function_configuration(
                FunctionName=resource_id,
                Environment={'Variables': spec.environment},
                MemorySize=spec.memory,
                Timeout=spec.timeout_seconds
            )

            resource.metadata.update({
                'memory_mb': spec.memory,
                'timeout_seconds': spec.timeout_seconds
            })
            resource.updated_at = datetime.now()

        return resource

    def delete_resource(self, resource_id: str) -> bool:
        """Delete a cloud resource"""
        resource = self.resources.get(resource_id)
        if not resource:
            return True

        try:
            if resource.resource_type == ResourceType.CONTAINER:
                ecs = self._get_client('ecs')

                # Scale down to 0
                ecs.update_service(
                    cluster=self.ecs_cluster,
                    service=resource_id,
                    desiredCount=0
                )

                # Delete service
                ecs.delete_service(
                    cluster=self.ecs_cluster,
                    service=resource_id
                )

            elif resource.resource_type == ResourceType.FUNCTION:
                lambda_client = self._get_client('lambda')
                lambda_client.delete_function(FunctionName=resource_id)

            elif resource.resource_type == ResourceType.STORAGE:
                s3 = self._get_client('s3')

                # Empty bucket first
                paginator = s3.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=resource_id)

                for page in pages:
                    if 'Contents' in page:
                        objects = [{'Key': obj['Key']} for obj in page['Contents']]
                        s3.delete_objects(
                            Bucket=resource_id,
                            Delete={'Objects': objects}
                        )

                # Delete bucket
                s3.delete_bucket(Bucket=resource_id)

            elif resource.resource_type == ResourceType.QUEUE:
                sqs = self._get_client('sqs')
                queue_url = resource.metadata['queue_url']
                sqs.delete_queue(QueueUrl=queue_url)

            elif resource.resource_type == ResourceType.SECRET:
                secrets = self._get_client('secretsmanager')
                secrets.delete_secret(
                    SecretId=resource_id,
                    ForceDeleteWithoutRecovery=True
                )

            # Remove from tracking
            del self.resources[resource_id]
            return True

        except Exception as e:
            logger.error(f"Error deleting resource {resource_id}: {e}")
            return False

    def list_resources(self, resource_type: Optional[ResourceType] = None) -> List[CloudResource]:
        """List all managed resources"""
        resources = list(self.resources.values())

        if resource_type:
            resources = [r for r in resources if r.resource_type == resource_type]

        return resources

    def estimate_cost(self, resources: List[CloudResource], days: int = 30) -> Dict[str, float]:
        """Estimate AWS costs"""
        cost_breakdown = {
            'compute': 0.0,
            'storage': 0.0,
            'network': 0.0,
            'other': 0.0,
            'total': 0.0
        }

        hours = days * 24

        for resource in resources:
            if resource.resource_type == ResourceType.CONTAINER:
                # ECS Fargate pricing
                cpu = float(resource.metadata.get('cpu', 1024)) / 1024
                memory = float(resource.metadata.get('memory', 2048)) / 1024

                cpu_cost = cpu * 0.04048 * hours
                memory_cost = memory * 0.004445 * hours

                if resource.metadata.get('launch_type') == 'FARGATE_SPOT':
                    # ~70% discount for spot
                    cpu_cost *= 0.3
                    memory_cost *= 0.3

                cost_breakdown['compute'] += cpu_cost + memory_cost

            elif resource.resource_type == ResourceType.FUNCTION:
                # Lambda pricing (estimates)
                memory_gb = resource.metadata.get('memory_mb', 512) / 1024
                requests_per_day = 100000  # Estimate
                duration_seconds = 0.1  # Estimate

                request_cost = (requests_per_day * days) * 0.0000002
                compute_cost = (requests_per_day * days * duration_seconds * memory_gb) * 0.0000166667

                cost_breakdown['compute'] += request_cost + compute_cost

            elif resource.resource_type == ResourceType.STORAGE:
                # S3 pricing (standard tier)
                storage_gb = 1000  # Estimate 1TB
                requests = 1000000  # Estimate 1M requests

                storage_cost = storage_gb * 0.023 * (days / 30)
                request_cost = requests * 0.0004 / 1000

                cost_breakdown['storage'] += storage_cost + request_cost

            elif resource.resource_type == ResourceType.QUEUE:
                # SQS pricing
                requests = 1000000  # Estimate 1M messages
                cost_breakdown['other'] += requests * 0.0000004

        # Add network transfer estimate (10GB/day)
        cost_breakdown['network'] = 10 * days * 0.09

        cost_breakdown['total'] = sum(
            cost_breakdown[k] for k in ['compute', 'storage', 'network', 'other']
        )

        return cost_breakdown

    # Helper methods
    def _get_subnets(self) -> List[str]:
        """Get default VPC subnets"""
        ec2 = self._get_client('ec2')

        response = ec2.describe_subnets(
            Filters=[
                {'Name': 'default-for-az', 'Values': ['true']}
            ]
        )

        return [subnet['SubnetId'] for subnet in response['Subnets']]

    def _get_security_group(self) -> str:
        """Get or create ops0 security group"""
        ec2 = self._get_client('ec2')

        group_name = 'ops0-default'

        try:
            response = ec2.describe_security_groups(
                GroupNames=[group_name]
            )
            return response['SecurityGroups'][0]['GroupId']

        except ec2.exceptions.ClientError:
            # Create security group
            response = ec2.create_security_group(
                GroupName=group_name,
                Description='Default security group for ops0 resources'
            )

            group_id = response['GroupId']

            # Add rules
            ec2.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 80,
                        'ToPort': 80,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    },
                    {
                        'IpProtocol': 'tcp',
                        'FromPort': 443,
                        'ToPort': 443,
                        'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                    }
                ]
            )

            return group_id

    def _get_lambda_role(self) -> str:
        """Get or create Lambda execution role"""
        iam = self._get_client('iam')

        role_name = 'ops0-lambda-execution-role'

        try:
            response = iam.get_role(RoleName=role_name)
            return response['Role']['Arn']

        except iam.exceptions.NoSuchEntityException:
            # Create role
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }

            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Execution role for ops0 Lambda functions'
            )

            # Attach policies
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )

            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess'
            )

            return response['Role']['Arn']

    def _get_ml_layer_arn(self) -> str:
        """Get ARN for ML dependencies layer"""
        # In production, this would reference a pre-built layer
        # with numpy, pandas, scikit-learn, etc.
        return f"arn:aws:lambda:{self.region}:{self.account_id}:layer:ops0-ml-deps:1"

    def _package_lambda_code(self, spec: DeploymentSpec) -> Dict[str, Any]:
        """Package code for Lambda deployment"""
        # In production, this would:
        # 1. Extract code from container image
        # 2. Create deployment package
        # 3. Upload to S3
        # 4. Return S3 location

        return {
            'ZipFile': b'dummy-code-package',  # Placeholder
            # 'S3Bucket': 'ops0-lambda-code',
            # 'S3Key': f'{spec.step_name}/code.zip'
        }

    def _setup_auto_scaling(self, service_name: str, spec: DeploymentSpec):
        """Configure ECS auto-scaling"""
        autoscaling = self._get_client('application-autoscaling')

        resource_id = f"service/{self.ecs_cluster}/{service_name}"

        # Register scalable target
        autoscaling.register_scalable_target(
            ServiceNamespace='ecs',
            ResourceId=resource_id,
            ScalableDimension='ecs:service:DesiredCount',
            MinCapacity=spec.min_instances,
            MaxCapacity=spec.max_instances
        )

        # Create scaling policy
        autoscaling.put_scaling_policy(
            PolicyName=f'{service_name}-cpu-scaling',
            ServiceNamespace='ecs',
            ResourceId=resource_id,
            ScalableDimension='ecs:service:DesiredCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': spec.target_cpu_percent,
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'ECSServiceAverageCPUUtilization'
                },
                'ScaleInCooldown': spec.scale_down_cooldown,
                'ScaleOutCooldown': spec.scale_up_cooldown
            }
        )