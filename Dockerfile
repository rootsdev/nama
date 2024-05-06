FROM python:3.13.0a6-slim
EXPOSE 8080
WORKDIR /code
COPY ./setup.py /code/setup.py
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN mkdir -p data/models
RUN curl https://nama-data.s3.amazonaws.com/data/models/fs-given-swivel-vocab-600000.csv --output data/models/fs-given-swivel-vocab-600000.csv
RUN curl https://nama-data.s3.amazonaws.com/data/models/fs-given-swivel-model-600000-100.pth --output data/models/fs-given-swivel-model-600000-100.pth
RUN curl https://nama-data.s3.amazonaws.com/data/models/fs-given-encoder-model-600000-100.pth --output data/models/fs-given-encoder-model-600000-100.pth
RUN curl https://nama-data.s3.amazonaws.com/data/models/fs-given-clusters-600000-100.csv.gz --output data/models/fs-given-clusters-600000-100.csv.gz
RUN curl https://nama-data.s3.amazonaws.com/data/models/fs-surname-swivel-vocab-2100000.csv --output data/models/fs-surname-swivel-vocab-2100000.csv
RUN curl https://nama-data.s3.amazonaws.com/data/models/fs-surname-swivel-model-2100000-100.pth --output data/models/fs-surname-swivel-model-2100000-100.pth
RUN curl https://nama-data.s3.amazonaws.com/data/models/fs-surname-encoder-model-2100000-100.pth --output data/models/fs-surname-encoder-model-2100000-100.pth
RUN curl https://nama-data.s3.amazonaws.com/data/models/fs-surname-clusters-2100000-100.csv.gz --output data/models/fs-surname-clusters-2100000-100.csv.gz
COPY ./references /code/references
COPY ./src /code/src
CMD ["uvicorn", "src.server.server:app", "--host", "0.0.0.0", "--port", "8080"]
