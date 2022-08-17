from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import json
import re
import boto3
from botocore.exceptions import ClientError
import tarfile
import spacy
nlp = spacy.load('en_core_web_md')


class Predict():
    def __init__(self, model_path, encoder_path, branch_name):
        self.model_dl = self.download_model(branch_name)
        self.model = self.read_model(model_path)
        self.encoder = self.read_encoder(encoder_path)

    def download_model(self, branch_name):
        ''' 
        Function to download model tar file from s3

            Arguments:
               * path: path to the model
            Outputs:
               * mo
        '''
        s3_client = boto3.client('s3')
        s3_client.download_file('pipeline-tcl-ver1',
                                f'model/{branch_name}/model.tar.gz',
                                'models/model.tar.gz')
        file = tarfile.open('models/model.tar.gz')
        file.extractall('models/')
        file.close()
        return True

    def read_model(self, model_path):
        ''' 
        Function to load a joblib file from a local file

            Arguments:
                * path: path to the model
            Outputs:
               * model: loaded joblib model 
        '''
        with open(model_path, 'rb') as f:
            self.model = joblib.load(f)
        return self.model

    def read_encoder(self, encoder_path):
        ''' 
           Function to load a the saved sklearn labelencoder

            Arguments:
               * path: path to the model
            Outputs:
               * model: loaded joblib model 
        '''
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.load(encoder_path, allow_pickle=True)
        return self.encoder

    @staticmethod
    def clean_text(words, remove_list=None):
        '''
        Function to clean text data

            Arguments:
                * remove_list: a list of words that should be removed
            Outputs:
                * cleaned text 
        '''
        if not words:
            return None

        if remove_list is None:

            remove_list = ["and", "to", "was", "it", "<<<", "into",
                           "a", "i", "with", 'that', ">>>",
                           "would", "with", "of", "[", "]",
                           "in", "he", "on", "*", "n", "at",
                           "for", "we", "y", "had", "or", "ip",
                           "from", "were", "is", "has", "her", "she",
                           "as", "this", "be", "his", "all", "my", "by", "any", "each", "will", 'a', 's', 'its', 'if', 'are']

        remove_list = remove_list + \
            re.findall(r"<<<\[([^\]\[\r\n]*)\]>>>", words)

        words = str(words)
        words = words.lower()
        words = words.replace("<<<", "")
        words = words.replace(">>>", "")
        words = words.replace("[", "")
        words = words.replace("]", "")
        words = words.replace("(", "")
        words = words.replace(")", "")
        words = words.replace("*", "")
        words = words.replace("what's", "what is ")
        words = words.replace(".", "")
        words = words.replace(",", "")
        words = words.replace("wasn't", "was not")
        words = words.replace("can't", "cannot")
        words = words.replace("it's", "it is")
        words = words.replace("it's", "it is")
        words = words.replace("\'ve", " have ")
        words = words.replace("'s", " ")
        words = words.replace("can't", "cannot ")
        words = words.replace("don't", "do not ")
        words = words.replace("doesn't", "does not ")
        words = words.replace("n't", "not ")
        words = words.replace(r"i'm", "i am")
        words = words.replace(r" m ", " am ")
        words = words.replace(r"\'re", " are ")
        words = words.replace(r"\'d", " would ")
        words = words.replace("'", "")
        words = words.replace("!", "")
        words = words.replace(r"\'ll", " will ")
        words = words.replace(r"´", "")
        words = words.replace("`", "")
        words = words.replace("\r\n", " ")
        words = words.replace("the", "")

        split_out = [x for x in words.split() if x not in remove_list]

        return " ".join(split_out)

    @staticmethod
    def get_embeddings(text):
        ''' 
        Function to maps the training dataset to the spacy embeddings
        Arguments:
            * target text
        Outputs:
            * list with embeddings
        '''
        return nlp(text).vector.reshape(1, -1)


def get_secret(secret_name, secret_key):
    """
    Function to get secret from aws.
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
