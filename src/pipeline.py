from settings import Settings
from statistics import mode
import tarfile
import sys
import logging
import os
import boto3
from datetime import datetime
import sagemaker
from sagemaker.processing import (ScriptProcessor,
                                  ProcessingInput,
                                  ProcessingOutput)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.workflow.steps import (ProcessingStep,
                                      TrainingStep,
                                      TuningStep)
from sagemaker.workflow.parameters import (ParameterInteger,
                                           ParameterString,)
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

from sagemaker.tuner import (CategoricalParameter,
                             HyperparameterTuner,
                             )

from tools.utils import (get_session,
                         get_bucket_name,
                         get_secret,
                         update_py_file,
                         push_image_ecr,
                         push_image_ecr_deployment,
                         )

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def get_init_data():
    """
    Function to return all the init data
    Outputs:
        * account_id: string
            aws account ID
        * region: string
            region to use
        * sagemaker_session: sagemekr.Session

        * branch_name: string
            branch name
    """
    bucket_name = get_bucket_name(settings)
    # bucket_name = settings.BUCKET_NAME
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    region = boto3.Session().region_name
    sagemaker_session = get_session(region, bucket_name)
    branch_name = sys.argv[1]
    role = get_secret('MLOps', 'EXECUTION_ROLE')
    if branch_name in ['develop', 'staging', 'production']:
        s3_prefix = branch_name
    else:
        s3_prefix = f'test-{branch_name}'
    return account_id, region, sagemaker_session, branch_name, s3_prefix, bucket_name, role


def data_processing(processing_repository_uri,
                    bucket_name,
                    branch_name,
                    role,
                    sagemaker_session,
                    processing_instance_count,
                    processing_instance_type):
    """
    """
    now = datetime.now().strftime(
        "%Y-%m-%d")  # get time stamp to be used for folder namers
    output_data_destination = f's3://{bucket_name}/data/{branch_name}/{now}'
    script_processor = ScriptProcessor(command=['python3'],
                                       image_uri=processing_repository_uri,
                                       role=role,
                                       instance_count=processing_instance_count,
                                       instance_type=processing_instance_type,
                                       sagemaker_session=sagemaker_session)

    step_preprocess_data = ProcessingStep(
        name="Data-Processing",
        processor=script_processor,
        inputs=[ProcessingInput(
            source=f's3://{bucket_name}/data/ledgar.parquet.gzip',
            destination='/opt/ml/processing/input/data')],
        outputs=[ProcessingOutput(
            output_name="df",
            source='/opt/ml/processing/output',
            destination=output_data_destination,)],
        code="src/data_processing/preprocessing.py",
    )
    return step_preprocess_data


def model_training(repository_uri,
                   bucket_name,
                   role,
                   sagemaker_session,
                   processing_instance_count,
                   processing_instance_type,
                   step_preprocess_data):
    """
    """
    now = datetime.now().strftime(
        "%Y-%m-%d")  # get time stamp to be used for folder namers
    estimator = Estimator(
        image_uri=repository_uri,
        role=role,
        instance_count=processing_instance_count,
        instance_type='ml.m5.large',
        sagemaker_session=sagemaker_session,
        output_path=f's3://{bucket_name}/model/pipelines/{now}/',
        disable_profiler=True,
        use_spot_instances=True,
        max_wait=90000,)

    hyperparameter_ranges = {
        "n_estimators": CategoricalParameter([100, 200, 300, 400]),
        "min_samples_split": CategoricalParameter([2, 4, 8]),
    }
    objective_metric_name = "accuracy test"
    objective_type = "Maximize"
    metric_definitions = [{"Name": "accuracy test",
                           "Regex": "Test set accuracy: ([0-9\\.]+)"}]

    tuner = HyperparameterTuner(
        estimator,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions,
        max_jobs=4,
        max_parallel_jobs=4,
        objective_type=objective_type,)

    step_tune_model = TuningStep(name="Hyperparameter-Optimization",
                                 tuner=tuner,
                                 inputs={
                                      "train": TrainingInput(
                                          s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                                              "df"].S3Output.S3Uri,
                                      ),
                                 },
                                 )

    step_train_model = TrainingStep(
        name="Train-Model",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                    "df"].S3Output.S3Uri,
            ),
        },
    )

    return step_train_model, step_tune_model


def model_deployment(repository_uri,
                     model_deployment_server_repository_uri,
                     bucket_name,
                     role,
                     sagemaker_session,
                     processing_instance_count,
                     processing_instance_type,
                     step_preprocess_train,
                     s3_prefix,
                     step_tune_model):
    """
    """
    now = datetime.now().strftime(
        "%Y-%m-%d")  # get time stamp to be used for folder namers
    model_path = f'{bucket_name}/model/pipelines/{now}'

    tar = tarfile.open(
        f'docker.tar.gzip', "w:bz2")
    for name in ["src/model_deployment"]:
        tar.add(name)
    tar.close()

    script_processor = ScriptProcessor(command=['python3'],
                                       image_uri=repository_uri,
                                       role=role,
                                       instance_count=processing_instance_count,
                                       sagemaker_session=sagemaker_session,
                                       instance_type=processing_instance_type)

    step_deployment = ProcessingStep(
        name="Model-Deployment",
        processor=script_processor,
        inputs=[
            ProcessingInput(source=step_tune_model.get_top_model_s3_uri(
                top_k=0,
                s3_bucket=model_path),
                # source=step_preprocess_train.properties.ModelArtifacts.S3ModelArtifacts,
                # source=f"s3://{bucket_name}/sagemaker/pipelines-fubxsd6smcxp-Train-Models-K9cOVsLX0P/output/model.tar.gz",
                destination="/opt/ml/processing/model",
            ),
        ],
        code="src/model_deployment/deployment.py",
    )
    return step_deployment


def model_evaluation(processing_repository_uri,
                     bucket_name,
                     branch_name,
                     role,
                     sagemaker_session,
                     processing_instance_count,
                     processing_instance_type,
                     step_preprocess_data,
                     step_preprocess_train,
                     step_tune_model,):
    """
    """
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json")

    now = datetime.now().strftime(
        "%Y-%m-%d")  # get time stamp to be used for folder namers
    output_data_destination = f's3://{bucket_name}/data/{branch_name}/{now}'
    model_path = f'{bucket_name}/model/pipelines/{now}'

    script_processor = ScriptProcessor(command=['python3'],
                                       image_uri=processing_repository_uri,
                                       role=role,
                                       sagemaker_session=sagemaker_session,
                                       instance_count=processing_instance_count,
                                       instance_type=processing_instance_type,
                                       )

    step_evaluation = ProcessingStep(
        name="Model-Evaluation",
        processor=script_processor,
        inputs=[
            ProcessingInput(
                source=step_preprocess_data.properties.ProcessingOutputConfig.Outputs[
                    "df"].S3Output.S3Uri,
                destination='/opt/ml/processing/input/data'),
            ProcessingInput(
                source=step_tune_model.get_top_model_s3_uri(
                    top_k=0,
                    s3_bucket=model_path),

                # source=step_preprocess_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"),
        ],
        outputs=[ProcessingOutput(
            output_name="evaluation",
            source='/opt/ml/processing/evaluation',
            destination=output_data_destination,)],
        code="src/model_evaluation/evaluate.py",
        property_files=[evaluation_report],
    )
    return step_evaluation, evaluation_report


def sagemaker_pipeline(bucket_name,
                       branch_name,
                       processing_repository_uri,
                       model_training_repository_uri,
                       model_deployment_repository_uri,
                       model_deployment_server_repository_uri,
                       role,
                       sagemaker_session,
                       pipeline_name,
                       s3_prefix):
    """
    Function for run a processing job
    """
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.4xlarge")

    step_preprocess_data = data_processing(processing_repository_uri=processing_repository_uri,
                                           bucket_name=bucket_name,
                                           branch_name=branch_name,
                                           role=role,
                                           sagemaker_session=sagemaker_session,
                                           processing_instance_count=processing_instance_count,
                                           processing_instance_type=processing_instance_type,
                                           )

    logger.info('step_preprocess_data added')

    step_preprocess_train, step_tune_model = model_training(repository_uri=model_training_repository_uri,
                                                            bucket_name=bucket_name,
                                                            role=role,
                                                            sagemaker_session=sagemaker_session,
                                                            processing_instance_count=processing_instance_count,
                                                            processing_instance_type=processing_instance_type,
                                                            step_preprocess_data=step_preprocess_data)

    logger.info('step_preprocess_train added')

    step_model_deployment = model_deployment(repository_uri=model_deployment_repository_uri,
                                             model_deployment_server_repository_uri=model_deployment_server_repository_uri,
                                             bucket_name=bucket_name,
                                             role=role,
                                             sagemaker_session=sagemaker_session,
                                             processing_instance_count=processing_instance_count,
                                             processing_instance_type=processing_instance_type,
                                             step_preprocess_train=step_preprocess_train,
                                             s3_prefix=s3_prefix,
                                             step_tune_model=step_tune_model)
    logger.info('step_model_deployment added')

    step_model_evaluation, evaluation_report = model_evaluation(processing_repository_uri=processing_repository_uri,
                                                                bucket_name=bucket_name,
                                                                branch_name=branch_name,
                                                                role=role,
                                                                sagemaker_session=sagemaker_session,
                                                                processing_instance_count=processing_instance_count,
                                                                processing_instance_type=processing_instance_type,
                                                                step_preprocess_data=step_preprocess_data,
                                                                step_preprocess_train=step_preprocess_train,
                                                                step_tune_model=step_tune_model)
    logger.info('step_model_evaluation step added')
    logger.info('evaluation_report step added')

    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_model_evaluation.name,
            property_file=evaluation_report,
            json_path="classification_metrics.accuracy.value",
        ),
        right=0.7,
    )

    # Create a Sagemaker Pipelines ConditionStep, using the condition above.
    # Enter the steps to perform if the condition returns True / False.
    step_cond = ConditionStep(
        name="Accuracy-Condition",
        conditions=[cond_gte],
        if_steps=[step_model_deployment],
        else_steps=[],
    )
    pipeline = Pipeline(name=pipeline_name,
                        parameters=[
                            processing_instance_type,
                            processing_instance_count],
                        steps=[step_preprocess_data, step_tune_model, step_model_evaluation, step_cond])
    # steps=[step_model_deployment],)
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    execution.wait()
    os.remove('docker.tar.gzip')
    return True


def main(pipeline_name):
    """
    Function to run the pipeline
    """
    account_id, region, sagemaker_session, branch_name, s3_prefix, bucket_name, role = get_init_data()

    update_py_file(s3_folder_prefix=s3_prefix,
                   original_file_path='src/model_deployment/predictor_base.py',
                   deploy_file_path='src/model_deployment/container/src/predictor.py')
    logger.info(
        f's3_folder_prefix in predictor_base.py updated with "{s3_prefix}" ')

    update_py_file(s3_folder_prefix=s3_prefix,
                   original_file_path='src/model_deployment/deployment_base.py',
                   deploy_file_path='src/model_deployment/deployment.py')
    logger.info(
        f's3_folder_prefix in deployment_base.py updated with "{s3_prefix}" ')

    push_image_ecr(docker_file_path=settings.DATA_PROCESSING_DOCKER,
                   tag_name='latest')
    logger.info('data processing image pushed to ECR successfully')
    push_image_ecr(docker_file_path=settings.MODEL_TRAINING_DOCKER,
                   tag_name='latest')
    logger.info('model training image pushed to ECR successfully')

    push_image_ecr(docker_file_path=settings.MODEL_DEPLOYMENT_DOCKER_SAGEMAKER,
                   tag_name='latest')
    logger.info('model deployment image pushed to ECR successfully')

    push_image_ecr(docker_file_path=settings.MODEL_DEPLOYMENT_DOCKER_SERVER,
                   tag_name='latest')
    logger.info('model deployment server image pushed to ECR successfully')

    processing_repository_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-processing-redj:{branch_name}'
    model_training_repository_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-train-redj:{branch_name}'
    model_deployment_repository_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-deployment-init-redj:{branch_name}'
    model_deployment_server_repository_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-deployment-redj:{branch_name}'

    sagemaker_pipeline(bucket_name=bucket_name,
                       branch_name=branch_name,
                       processing_repository_uri=processing_repository_uri,
                       model_training_repository_uri=model_training_repository_uri,
                       model_deployment_repository_uri=model_deployment_repository_uri,
                       model_deployment_server_repository_uri=model_deployment_server_repository_uri,
                       role=role,
                       sagemaker_session=sagemaker_session,
                       pipeline_name=pipeline_name,
                       s3_prefix=s3_prefix,
                       )

    # clean uploaded code artifacts to s3
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)

    folder_name_to_check = ['Data-Processing',
                            'Model-Deployment',
                            'Model-Evaluation']
    for my_bucket_object in bucket.objects.all():
        if any(ext in my_bucket_object.key for ext in folder_name_to_check):
            delete_folder_path = my_bucket_object.key
            delete_folder_path = delete_folder_path.split('/')[0]
            bucket.objects.filter(Prefix=delete_folder_path).delete()

    logger.info('pipeline ran successfully')


if __name__ == '__main__':
    settings = Settings()
    pipeline_name = settings.PIPELINE_NAME
    main(pipeline_name)
