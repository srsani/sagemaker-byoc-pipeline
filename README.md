# AWS BYOC pipeline for a text classification application

In the current code repository, you can find the code for the Amazon SageMaker MLOps workflow
for a Text classification application using Bring Your Own Container (BYOC) approach, which means each note in the pipeline is based on docker image that is stored in Elastic Container Registry (ECR).  This work is the final project for Udacity AWS MLOps nano degree.

Running this repo will create the following resources in AWS:

- One S3 bucket to download the raw file and store the data from the pipeline
- Fours ECR repositories
- Register a model in SageMaker
- Add a model config to SageMaker inference
- SageMaker pipeline
- SageMaker end-point

## Setup

### AWS side

1- setup SageMaker studio in `us-east-1`

2- make IAM `AmazonSageMaker-ExecutionRole` role with:

    - AmazonSageMaker-ExecutionPolicy
    - SecretsManagerReadWrite
    - AutoScalingFullAccess
    - AmazonS3FullAccess
    - AmazonSageMakerFullAccess

3- add the role that was just created to `AWS Secrets Manager`:

    - Make a MLOps secret group
![Alt text](images/1.png?raw=true "T")

    - Add the IAM role to `EXECUTION_ROLE`
![Alt text](images/2.png?raw=true "T")

### local setup

This repos is tested on python3.8:

- `virtualenv venv --python=python3.8`
- `pip install ipykernel`
- `python -m ipykernel install --user --name venv --display-name PYTHON_ENV_NAME`
- `pip install -r requirements.txt`

#### update code

Amend the prefix for all the ECR repositories form `-redj` to some other prefix

- `processing_repository_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-processing-redj:{branch_name}'`
- `model_training_repository_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-train-redj:{branch_name}'`
- `model_deployment_repository_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-deployment-init-redj:{branch_name}'`
- `model_deployment_server_repository_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-deployment-redj:{branch_name}'`

And all the corresponding docker .sh

- `src/data_processing/docker_ecr.sh`
- `src/model_deployment/docker_ecr.sh`
- `src/model_deployment/docker_ecr.sh`

## Run the pipeline

The following command is used to run the pipeline from your local env:

`python src/pipeline.py BRANCH_NAME`
