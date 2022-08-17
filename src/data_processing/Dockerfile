FROM python:3.8
COPY requirements.txt .
RUN pip3 install -r requirements.txt 
RUN python -m spacy download en_core_web_md
ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python3"]