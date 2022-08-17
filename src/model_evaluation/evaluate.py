import json
import logging
import glob
import pathlib
import pickle
import tarfile
import numpy as np
import pandas as pd
import joblib
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             confusion_matrix)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def make_model_ready_data(df):
    """
    Function for getting train val test data

    Arguments:
        Inputs:
             * df: panda panda dataframe
                input df
        Outputs:
            * X_train: panda dataframe
                 train_X
            * y_train: pandas series
                label for train
            * X_val: panda dataframe
                val_X
            * y_val: pandas series
                label for val
            * X_test: panda dataframe
                test_X
            * y_test: pandas series
                label for test           
    """
    train = df[df['data_type'] == 'train'].copy()
    val = df[df['data_type'] == 'val'].copy()
    test = df[df['data_type'] == 'test'].copy()

    X_train = pd.DataFrame(train.features.to_list())
    y_train = train.target_label

    X_val = pd.DataFrame(val.features.to_list())
    y_val = val.target_label

    X_test = pd.DataFrame(test.features.to_list())
    y_test = test.target_label

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_model(model_path):
    ''' 
    Function to load a joblib file from a local file

        Arguments:
            * path: path to the model
        Outputs:
           * model: loaded joblib model 
    '''
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model


def load_encoder(encoder_path,):
    '''
    Function to load a the saved sklearn labelencoder

        Arguments:
            * path: string
                path to the model
        Outputs:
            * sklearn encoder: sklean object
                loaded joblib model
    '''
    encoder = LabelEncoder()
    encoder.classes_ = np.load(encoder_path, allow_pickle=True)
    return encoder


def main(df_path, model_path):

   # extract model tar file
    with tarfile.open(model_path) as tar:
        tar.extractall()

    logger.info(glob.glob("*"))

    # load model
    model = load_model('model.joblib')
    # load encoder
    encoder = load_encoder('le_classes.npy')

    # load data
    df = pd.read_parquet(df_path)
    df['features'] = df.features.apply(lambda x: json.loads(x))
    df['target_label'] = df.target.apply(lambda x: encoder.transform([x])[0])
    X_train, y_train, X_val, y_val, X_test, y_test = make_model_ready_data(df)

    prediction_probabilities = model.predict(X_test)
    predictions = np.round(prediction_probabilities)

    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    dict_out = {}
    for i in range(len(conf_matrix)):
        dict_1 = {"0": int(conf_matrix[i][0]), "1": int(conf_matrix[i][1]), "1": int(conf_matrix[i][1]), "2": int(
            conf_matrix[i][2]), "3": int(conf_matrix[i][3]), "4": int(conf_matrix[i][4]), "5": int(conf_matrix[i][5])},
        dict_out[f"{i}"] = dict_1
    report_dict = {
        "classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "confusion_matrix": dict_out,
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": list([]),
                "true_positive_rates": list([]),
            },
        },
    }
    logger.info(report_dict)
    return report_dict


if __name__ == "__main__":
    # make folder path for evaluation
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_path = "/opt/ml/processing/input/data/df.parquet.gzip"

    logger.info('model folder:')
    logger.info(glob.glob("/opt/ml/processing/model/*"))
    logger.info('='*20)

    report_dict = main(test_path, model_path)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
