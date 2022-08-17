import os
from pathlib import Path
from pydantic import BaseSettings, AnyUrl
import uuid
import sys


class Settings(BaseSettings):
    sys.path.append('../')

    DEBUG = "1"
    DATA_PROCESSING_DOCKER = 'src/data_processing/docker_ecr.sh'
    MODEL_TRAINING_DOCKER = 'src/model_training/docker_ecr.sh'
    MODEL_DEPLOYMENT_DOCKER_SAGEMAKER = 'src/model_deployment/docker_ecr_init.sh'
    MODEL_DEPLOYMENT_DOCKER_SERVER = 'src/model_deployment/docker_ecr.sh'
    PIPELINE_NAME = 'pipeline-tcl'
    BUCKET_NAME = ''

    @staticmethod
    def get_src_dir():
        return Path(os.path.dirname(__file__))
