import subprocess
from pathlib import Path
import boto3
from sagemaker import get_execution_role
from sagemaker.model_monitor import DataCaptureConfig
from time import gmtime, strftime
import time
import json
import requests


def update_predictor(s3_folder_postfix):
    """
    Function for updating predictor.py with the current branch_name

    Arguments::
        * branch_name: string
            name of the current branch
    Outputs:
         * output: string
             result of pushed ecr
    """
    file_path = Path().absolute()
    original_file_path = Path(
        str(file_path)+'/model_deployment/predictor_base.py')
    deploy_file_path = Path(
        str(file_path)+'/model_deployment/container/src/predictor.py')

    f = open(original_file_path, 'r')
    linelist = f.readlines()
    f.close

    # Re-open file here
    f2 = open(deploy_file_path, 'w')
    for line in linelist:
        line = line.replace('s3_folder_postfix', s3_folder_postfix)
        f2.write(line)
    f2.close()
    return True


def push_image_ecr(ecr_prefix, model_artifact_path):
    """
    Function for getting initial data that is needed for the main function 

    Arguments::
        * ecr_prefix: string
            name for ecr repo
    Outputs:
         * output: string
             result of pushed ecr
    """
    file_path = Path().absolute()
    docker_file_path = Path(str(file_path) + '/model_deployment/docker_ecr.sh')
    bashCommand = f"chmod +x {docker_file_path}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"sh {docker_file_path} {ecr_prefix} {model_artifact_path}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output


def model_creation(s3_bucket, account_id, region, client, role, branch_name):
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
        * branch_name: string
            branch name that maches ecr path
    Outputs:
         * model_name: string
             new model name
    """
    time_stamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    model_name = f'{branch_name}-{time_stamp}'
    model_url = 's3://{}/src/'.format(s3_bucket)
    container = f'{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-hmlr-{branch_name}:latest'

    print(f'Model name: {model_name}')
    print(f'Model data Url: {model_url}')
    print(f'Container image: {container}')

    container = {
        'Image': container
    }

    create_model_response = client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        Containers=[container])

    print("Model Arn: " + create_model_response['ModelArn'])
    return model_name


def endpoint_config_creation(model_name, client):
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
    print(f'Endpoint config name: {endpoint_config_name} ')

    instance_type = 'ml.t2.medium'

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
            'DestinationS3Uri': 's3:/pipeline-tcl-ver1/data_capture/',
            'CaptureOptions': [
                {
                    'CaptureMode': 'Output'
                },
            ],
        }
    )

    end_point_arn = create_endpoint_config_response['EndpointConfigArn']
    print(f"Endpoint config Arn: {end_point_arn} ")
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
    print(f'Endpoint name: {endpoint_name}')

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
    print('Endpoint Arn: ' + create_endpoint_response['EndpointArn'])

    resp = client.describe_endpoint(EndpointName=endpoint_name)
    status = resp['EndpointStatus']
    print(f"Endpoint Status: {status}")

    print(f'Waiting for {endpoint_name} endpoint to be in service...')
    waiter = client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    return endpoint_name
