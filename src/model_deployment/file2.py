from flask import Flask
import flask
import spacy
import os
import json
import logging
import boto3
from utilities import Predict
import time

main = Predict('models/model.joblib','models/le_classes.npy' )
s3 = boto3.client('s3')

app = Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    
    health = main is not None
    status = 200 if health else 404
    return flask.Response(response= '\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
   
    input_json = flask.request.get_json()
    print(input_json)
    input_text = input_json['text_input']
    
    em = main.get_embeddings(input_text)
    predict_result = main.model.predict(em)
    model_result = str(predict_result[0])
    label = main.encoder.inverse_transform(predict_result)
    proba = str(main.model.predict_proba(em)[0])
    
    result = {
        'label_number': model_result,
        'label_name':label[0],
        'predict_proba': proba
        }
    
    #lod data to s3
    bucket_name='hmlr-dp-poc1-data'
    data_key = f'logs/test/enforce/{str(int(time.time()))}.json'

    s3.put_object(
         Body=json.dumps(result),
         Bucket=bucket_name,
         Key=data_key
    )
    
    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200, mimetype='application/json')