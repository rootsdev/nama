FROM python:3.8
EXPOSE 8080
WORKDIR /code
COPY ./setup.py /code/setup.py
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN mkdir -p data/models
RUN curl https://nama-data.s3.amazonaws.com/data/models/anc-triplet-bilstm-100-512-40-05.pth --output data/models/anc-triplet-bilstm-100-512-40-05.pth
RUN curl https://nama-data.s3.amazonaws.com/data/models/surname_clusters.tsv --output data/models/surname_clusters.tsv
COPY ./src /code/src
CMD ["uvicorn", "src.server.server:app", "--host", "0.0.0.0", "--port", "8080"]
