FROM python:3.8
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY train.py /opt/ml/code/train.py
ENV SAGEMAKER_PROGRAM train.py