import logging
from pathlib import Path
from io import BytesIO
import os
import boto3
from botocore.exceptions import ClientError
import json
import sagemaker.session
import sagemaker
import subprocess
import tarfile
import uuid

import pandas as pd
from datasets import (load_dataset, load_dataset_builder)


def add_init_data(bucket_name):
    """
    Function to download LEDGAR dataset HuggingFace and upload it to the project s3 bucket 

    Args:
        bucket_name (str): name of the working bucket

    Returns:
       True: file uploaded correctly
       False: anything else
    """
    try:
        dataset_builder = load_dataset_builder('lex_glue', 'ledgar')
        dataset_label = dataset_builder.info.features['label']

        train = load_dataset('lex_glue', 'ledgar', split='train').to_pandas()
        test = load_dataset('lex_glue', 'ledgar', split='test').to_pandas()
        validation = load_dataset('lex_glue', 'ledgar',
                                  split='validation').to_pandas()

        train['clause_type'] = train.label.apply(
            lambda x: dataset_label.int2str(x))
        test['clause_type'] = test.label.apply(
            lambda x: dataset_label.int2str(x))
        validation['clause_type'] = validation.label.apply(
            lambda x: dataset_label.int2str(x))
        df = pd.concat([train, test, validation])
        df.to_parquet(f's3://{bucket_name}/data/ledgar.parquet.gzip')
    except ClientError as e:
        print(str(e))
        return False
    return True


def get_bucket_name(settings):
    """
    get_bucket_name _summary_

    _extended_summary_

    Args:
        settings (_type_): _description_

    Returns:
        _type_: _description_
    """
    if settings.BUCKET_NAME != '':
        bucket_name = settings.BUCKET_NAME
    if settings.BUCKET_NAME == '':
        bucket_name_prefix = str(uuid.uuid4())
        bucket_name = f"pipeline-{bucket_name_prefix}"
        # create the bucket
        create_bucket(bucket_name)
        # read and change BUCKET_NAME in settings.py
        setting_file_path = 'src/settings.py'
        f1 = open(setting_file_path, "rt")
        data = f1.read()
        data = data.replace("BUCKET_NAME = \'\'\n",
                            f"BUCKET_NAME = \'{bucket_name}'\n")
        f1.close()
        # updated settings.py file
        f1 = open(setting_file_path, "wt")
        f1.write(data)
        f1.close()
        # adding ledgar data to the bucket
        add_init_data(bucket_name)
    return bucket_name


def create_bucket(bucket_name, region=None):
    """
    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).


    Args:
        bucket_name (str): bucket name to create
        region (str):  String region to create bucket in, e.g., 'us-west-2'

    Returns:
        True: bucket created
        False: bucket not created
    """

    # Create bucket
    try:
        if region is None:
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name,
                                    CreateBucketConfiguration=location)
    except ClientError as e:
        logging.error(e)
        return False
    return True


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
        service_name='secretsmanager')
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


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: string
            the aws region to start the session
        default_bucket: string
            the bucket to use for storing the artifacts
    Returns:
        sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def update_py_file(s3_folder_prefix, original_file_path, deploy_file_path):
    """
    Function for updating predictor.py with the current branch_name. 
    The base file need to have "s3_folder_prefix"

    Arguments::
        * branch_name: string
            name of the current branch
        * original_file_path: sting 
            path to the _base file
        * deploy_file_path: string
            path to where the file should go
    Outputs:
         * output: string
             result of pushed ecr
    """
    f1 = open(original_file_path, 'r')
    linelist = f1.readlines()

    # make a new file
    f2 = open(deploy_file_path, 'w')
    for line in linelist:
        line = line.replace('s3_folder_prefix', s3_folder_prefix)
        f2.write(line)
    f2.close()
    f1.close()
    return True


def push_image_ecr(docker_file_path, tag_name=None):
    """
    Function for getting pushing docker image to ecr

    Arguments::
        * ecr_prefix: string
            name for ecr repo
    Outputs:
         * output: string
             result of pushed ecr
    """
    bashCommand = f"chmod +x {docker_file_path}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"sh {docker_file_path} {tag_name}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output


def push_image_ecr_deployment(docker_file_path, tag_name, model_artifact_path):
    """
    Function for getting pushing docker image to ecr

    Arguments::
        * ecr_prefix: string
            name for ecr repo
    Outputs:
         * output: string
             result of pushed ecr
    """
    bashCommand = f"chmod +x {docker_file_path}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    bashCommand = f"sh {docker_file_path} {tag_name} {model_artifact_path}"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output


def make_deployment_tar():
    tar = tarfile.open(
        f'src/model_deployment/docker.tar.gzip', "w:bz2")
    for name in ["src/model_deployment"]:
        tar.add(name)
    tar.close()
    return True
