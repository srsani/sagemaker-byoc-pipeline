import glob
import tarfile
import subprocess
import logging
import sys
import boto3
from botocore.exceptions import ClientError
import json
from time import gmtime, strftime
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def get_secret(secret_name, secret_key):
    """
    Function to get secrest from aws.
    Arguments:
        * secret_name: string
            secret name on aws
    Outputs:
        * secret_value: string
    """
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1')
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            raise e
    else:
        secrets = json.loads(get_secret_value_response['SecretString'])
    return secrets[secret_key]


def uncompress_tar(file_path, target_dir=None):
    """
    Function for extract tar.gz file

    Arguments:
        * file_path: string
            path to the tar.gz
        * target_dir: string
            if None current folder
            else extract path
    Outputs:
        * secret_value: string
    """
    file = tarfile.open(file_path)
    if not target_dir:
        file.extractall('')
    else:
        file.extractall(target_dir)
    file.close()
    return True


def upload_data_s3():
    """
    Function to upload model.tar.gz to a s3 bucket
    """
    s3_client = boto3.client('s3')
    now = datetime.now().strftime("%Y-%m-%d")
    try:
        response = s3_client.upload_file('/opt/ml/processing/model/model.tar.gz',
                                         bucket_name,
                                         f'model/develop/model.tar.gz')
    except ClientError as e:
        raise e
    return response


def model_creation(s3_bucket, account_id, region, client, role, branch_name, container_uri):
    """
    Function creating a new model 

    Arguments::
        * s3_bucket: string
            path for s3 bucket
        * account_id: string
            aws account id
        * region: sting
            region name
        * client: botocore.client.SageMaker
            boto3 instance connected to SageMaker
        * role: string
            iam role with execution permission
        * branch_name: stirng
            branch name that maches ecr path
    Outputs:
         * model_name: string
             new model name
    """
    time_stamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    model_name = f'{branch_name}-{time_stamp}'
    model_url = 's3://{}/src/'.format(s3_bucket)

    logger.info(f'Model name: {model_name}')
    logger.info(f'Model data Url: {model_url}')
    logger.info(f'Container image: {container_uri}')

    container = {
        'Image': container_uri
    }

    create_model_response = client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        Containers=[container])

    logger.info("Model Arn: " + create_model_response['ModelArn'])
    return model_name


def endpoint_config_creation(model_name, client, bucket_name):
    """
    Function creating a new endpoint config file 

    Arguments:
        * model_name: sting
           model name
        * client: botocore.client.SageMaker
            boto3 instance connected to SageMaker  
    Outputs:
         * end_point_arn: string
             end point arn 
         * endpoint_config_name: string
             generated en
    """
    endpoint_config_name = 'poc-ver-one-config' + \
        strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    logger.info(f'Endpoint config name: {endpoint_config_name} ')

    instance_type = 'ml.c5.large'

    create_endpoint_config_response = client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'InstanceType': instance_type,
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 1,
            'ModelName': model_name,
            'VariantName': 'AllTraffic'}],
        DataCaptureConfig={
            'EnableCapture': True,
            'InitialSamplingPercentage': 100,
            'DestinationS3Uri': f's3://{bucket_name}/data_capture/develop',
            'CaptureOptions': [
                {
                    'CaptureMode': 'Output'
                },
            ],
        }
    )

    end_point_arn = create_endpoint_config_response['EndpointConfigArn']
    logger.info(f"Endpoint config Arn: {end_point_arn} ")
    return end_point_arn, endpoint_config_name


def endpoint_creation(endpoint_config_name, client, endpoint_name):
    """
    Function creating a new endpoint config file 

    Arguments::
        * model_name: sting
           model name
        * client: botocore.client.SageMaker
            boto3 instance connected to SageMaker  
    Outputs:
         * endpoint_name: string
             generated endpoint name
    """
    str_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    logger.info(f'Endpoint name: {endpoint_name}')

    try:
        create_endpoint_response = client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name)
    except:

        create_endpoint_response = client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name)
    else:
        pass
#         client.delete_endpoint(EndpointName=endpoint_name)
#         client.delete_endpoint_config(EndpointConfigName = endpoint_config_name)
#         create_endpoint_response = client.create_endpoint(
#                                                     EndpointName = endpoint_name,
#                                                     EndpointConfigName = endpoint_config_name)
    logger.info('Endpoint Arn: ' + create_endpoint_response['EndpointArn'])

    resp = client.describe_endpoint(EndpointName=endpoint_name)
    status = resp['EndpointStatus']
    logger.info(f"Endpoint Status: {status}")

    logger.info(f'Waiting for {endpoint_name} endpoint to be in service...')
    waiter = client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    return endpoint_name


def endpoint_autoscaling(endpoint_name):
    asg = boto3.client(service_name='application-autoscaling',
                       region_name='us-east-1')

    # Resource type is variant and the unique identifier is the resource ID.
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    # scaling configuration
    response = asg.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=1,
        MaxCapacity=4
    )
    # CPUUtilization metric
    response = asg.put_scaling_policy(
        PolicyName='CPUUtil-ScalingPolicy',
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': 90.0,
            'CustomizedMetricSpecification':
            {
                'MetricName': 'CPUUtilization',
                'Namespace': '/aws/sagemaker/Endpoints',
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'VariantName', 'Value': 'AllTraffic'}
                ],
                # Possible - 'Statistic': 'Average'|'Minimum'|'Maximum'|'SampleCount'|'Sum'
                'Statistic': 'Average',
                'Unit': 'Percent'
            },
            'ScaleInCooldown': 600,
            'ScaleOutCooldown': 300
        }
    )

    return True


def init_req_info():
    """
    Function for getting initial data that is needed for the main function 

    Outputs:
        * bucket_name: string
            buckat name
        * branch_name: string
            working branch name
        * role: string     
    """
    bucket_name = get_secret('MLOps', 'BUCKET_NAME')
    account_id = boto3.client('sts').get_caller_identity()['Account']
    role = get_secret('MLOps', 'EXECUTION_ROLE')
    # region = boto3.Session().region_name
    region = 'us-east-1'
    return bucket_name, account_id, role, region


if __name__ == "__main__":

    client = boto3.client(service_name='sagemaker',
                          region_name='us-east-1')
    bucket_name, account_id, role, region = init_req_info()
    container_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-deployment-redj:develop'

    uncompress_tar('/opt/ml/processing/model/model.tar.gz')

    logger.info('/opt/ml/processing/model')
    logger.info(glob.glob("/opt/ml/processing/model/*"))
    logger.info(glob.glob("*"))

    upload_data_s3()
    logger.info('='*20)
    logger.info('upload model tar file to model/develop/')

    model_name = model_creation(s3_bucket=bucket_name,
                                account_id=account_id,
                                region=region,
                                client=client,
                                role=role,
                                branch_name='develop',
                                container_uri=container_uri)
    logger.info(model_name)

    end_point_arn, endpoint_config_name = endpoint_config_creation(model_name=model_name,
                                                                   client=client,
                                                                   bucket_name=bucket_name)

    endpoint_name = endpoint_creation(endpoint_config_name=endpoint_config_name,
                                      client=client,
                                      endpoint_name='develop')

    endpoint_autoscaling(endpoint_name)
