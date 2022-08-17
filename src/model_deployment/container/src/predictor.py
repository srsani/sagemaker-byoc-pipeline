from flask import Flask
import flask
import spacy
import os
import json
import logging
import boto3
from utilities import (Predict, get_secret)
import time
import uuid

main = Predict('models/model.joblib',
               'models/le_classes.npy',
               'develop')
s3 = boto3.client('s3')
bucket_name = get_secret('MLOps', 'BUCKET_NAME')
app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():

    health = main is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():

    input_json = flask.request.get_json()
    # print(input_json)
    input_text = input_json['text_input']

    em = main.get_embeddings(input_text)
    predict_result = main.model.predict(em)
    model_result = str(predict_result[0])
    label = main.encoder.inverse_transform(predict_result)
    proba = main.model.predict_proba(em)[0].tolist()

    result = {
        'label_number': model_result,
        'label_name': label[0],
        'predict_proba': proba
    }

    json_object = {
        'metadata': {
            'source': 'develop',
            'type': 'prediction',
            'datetime':  int(time.time()),

        },
        'payload': {
            'input_text': input_json['text_input'],
            'features': em.tolist(),
            'inference_result_proba': proba,
            'inference_result_label': label[0],
            'input_json': input_json['text_input'],
            'label_number': model_result,
            'label_name': label[0]
        }
    }

    # load data to s3
    # bucket_name = get_secret('MLOps', 'BUCKET_NAME')
    log_name = str(uuid.uuid4())
    data_key = f'logs/develop/{log_name}.json'

    s3.put_object(
        Body=json.dumps(json_object),
        Bucket=bucket_name,
        Key=data_key
    )

    # resultjson = json.dumps(result)
    resultjson = json.dumps(json_object)
    return flask.Response(response=resultjson, status=200, mimetype='application/json')
