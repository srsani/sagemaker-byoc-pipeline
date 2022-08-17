from datetime import datetime
from model_deployment.api_deployment import *
from tools.utils import *
import sys
import os
os.path.abspath(os.path.join('..'))


def init_req_info():
    """
    Function for getting initial data that is needed for the main function 

    Outputs:
        * bucket_name: string
            buckat name
        * branch_name: string
            working branch name
        * s3_folder_postfix: string     
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d--%H-%M-%S")
    file_name = f'{os.path.basename(__file__)[:-3]}--{date_time}'
    branch_name, s3_folder_postfix = get_active_branch_name()
    bucket_name = 'hmlr-dp-poc1-data'
    return bucket_name, branch_name, s3_folder_postfix, file_name


def model_deployment_main(model_artifact_path):
    """
    Function that deploys a model to sagemaker

    Arguments:
        * endpoint_name: string
            endpoint name
    Outputs:
        * endpoint_name: string
            new endpoint name
    """
    ##
    bucket_name, branch_name, s3_folder_postfix, file_name = init_req_info()

    client = boto3.client(service_name='sagemaker')
    account_id = boto3.client('sts').get_caller_identity()['Account']
    region = boto3.Session().region_name
    # not used here
    s3_bucket = 'hmlr-dp-poc1-data'
    role = get_execution_role()
    print(bucket_name, branch_name, s3_folder_postfix, file_name)

    # update log path in the predictor.py
    update_predictor(s3_folder_postfix)
    # make a new docker image and push it to ecr
    push_image_ecr(ecr_prefix=branch_name,
                   model_artifact_path=model_artifact_path)
    # make a new model
    model_name = model_creation(s3_bucket=s3_bucket,
                                account_id=account_id,
                                region=region,
                                client=client,
                                role=role,
                                branch_name=branch_name)
    # make a new endpoint config
    end_point_arn, endpoint_config_name = endpoint_config_creation(model_name=model_name,
                                                                   client=client)
    # deploy a new andpoint
    endpoint_name = endpoint_creation(endpoint_config_name=endpoint_config_name,
                                      client=client,
                                      endpoint_name=branch_name)
    return endpoint_name
