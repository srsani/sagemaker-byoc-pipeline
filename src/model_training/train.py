import argparse
import pandas as pd
import numpy as np
import glob
import pathlib
import json
import boto3
import os
import joblib
import logging
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

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
                label for trian
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


def main(df,
         n_estimators,
         min_samples_split):

    X_train, y_train, X_val, y_val, X_test, y_test = make_model_ready_data(df)
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   min_samples_split=int(min_samples_split),
                                   n_jobs=-1)
    model.fit(X_train, y_train)
    X = pd.DataFrame(df.features.to_list())
    df['y_pred'] = model.predict(X)
    y_test_pred = model.predict(X_test)
    # accuracy = model.score(X_test, y_test)
    accuracy = f1_score(y_test, y_test_pred, average='weighted')
    logger.info(f"\nTest set accuracy: {accuracy} \n")
    df.to_parquet('/opt/ml/model/df.parquet.gzip', compression='gzip')

    return model


if __name__ == "__main__":
    pathlib.Path('/opt/ml/model').mkdir(parents=True, exist_ok=True)
    pathlib.Path('/opt/ml/input/data/train').mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="RandomForestClassifier Example")
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=400,
        metavar="N",
        help="n_estimators",
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        metavar="min_samples_split",
        help="min_samples_split",
    )
    args = parser.parse_args()
    n_estimators = args.n_estimators
    min_samples_split = args.min_samples_split
    logger.info(glob.glob("*"))

    df = pd.read_parquet('/opt/ml/input/data/train/df.parquet.gzip')
    df['features'] = df.features.apply(lambda x: json.loads(x))
    label_encoder = LabelEncoder()
    df['target_label'] = label_encoder.fit_transform(df.target)

    model = main(df, n_estimators, min_samples_split)
    file_name = 'model.tar.gzip'
    object_name = os.path.join('test', os.path.basename(file_name))

    with open('model.joblib', 'wb') as f:
        joblib.dump(model, f)
    np.save('le_classes.npy', label_encoder.classes_)

    with open('/opt/ml/model/model.joblib', 'wb') as f:
        joblib.dump(model, f)
    np.save('/opt/ml/model/le_classes.npy', label_encoder.classes_)
    logger.info(glob.glob("*"))
