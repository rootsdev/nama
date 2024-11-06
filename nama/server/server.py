import time
from collections import namedtuple
from enum import Enum

import pandas as pd
import torch
from fastapi import FastAPI
from pydantic import BaseModel, validator

from nama.data.normalize import normalize, standardize_patronymics
from nama.data.utils import load_nicknames
from nama.models.cluster import get_clusters, merge_name2clusters, read_clusters
from nama.models.swivel import SwivelModel, get_swivel_embeddings
from nama.models.utils import add_padding

# run using uvicorn src.server.server:app --reload

embed_dim = 100
given_vocab_size = 600000
surname_vocab_size = 2100000
prefix = "data/"  # s3://nama-data/data/
given_swivel_vocab_path = f"{prefix}models/fs-given-swivel-vocab-{given_vocab_size}.csv"
given_swivel_model_path = f"{prefix}models/fs-given-swivel-model-{given_vocab_size}-{embed_dim}.pth"
given_encoder_model_path = f"{prefix}models/fs-given-encoder-model-{given_vocab_size}-{embed_dim}.pth"
given_clusters_path = f"{prefix}models/fs-given-clusters-{given_vocab_size}-{embed_dim}.csv.gz"
surname_swivel_vocab_path = f"{prefix}models/fs-surname-swivel-vocab-{surname_vocab_size}.csv"
surname_swivel_model_path = f"{prefix}models/fs-surname-swivel-model-{surname_vocab_size}-{embed_dim}.pth"
surname_encoder_model_path = f"{prefix}models/fs-surname-encoder-model-{surname_vocab_size}-{embed_dim}.pth"
surname_clusters_path = f"{prefix}models/fs-surname-clusters-{surname_vocab_size}-{embed_dim}.csv.gz"
nicknames_path = "references/givenname_nicknames.csv"
max_search_clusters = 25

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# force models to the cpu
device = "cpu"


def load_swivel_vocab(path):
    with open(path, "rb") as f:
        swivel_vocab_df = pd.read_csv(f)
    return swivel_vocab_df.set_index("name")["index"].to_dict()


def load_swivel_model(path, vocab_len):
    swivel_model = SwivelModel(vocab_len, embedding_dim=embed_dim)
    with open(path, "rb") as f:
        swivel_model.load_state_dict(torch.load(f, map_location=torch.device(device)))
    swivel_model.to(device)
    swivel_model.eval()
    return swivel_model


def load_encoder_model(path):
    # TODO re-load encoder model when ready
    encoder_model = None
    # encoder_model = SwivelEncoderModel(output_dim=embed_dim, device=device)
    # encoder_model.load_state_dict(torch.load(open(path, "rb"), map_location=torch.device(device)))
    # encoder_model.to(device)
    # encoder_model.eval()
    return encoder_model


Clusters = namedtuple("Clusters", "clustered_name_cluster_ids clustered_name_embeddings")


def load_clusters(path, swivel_vocab, swivel_model, encoder_model):
    name2cluster = read_clusters(path)
    clustered_names = list(name2cluster.keys())
    clustered_name_cluster_ids = list(name2cluster.values())
    clustered_name_embeddings = get_swivel_embeddings(
        model=swivel_model, vocab=swivel_vocab, names=clustered_names, encoder_model=encoder_model
    )
    return Clusters(
        clustered_name_cluster_ids=clustered_name_cluster_ids, clustered_name_embeddings=clustered_name_embeddings
    )


start_time = time.time()

print("..loading vocabs", end="")
task_start_time = time.time()
swivel_vocab = {
    "given": load_swivel_vocab(given_swivel_vocab_path),
    "surname": load_swivel_vocab(surname_swivel_vocab_path),
}
task_end_time = time.time()
print(f" {task_end_time - task_start_time} seconds")  # memory=", get_size(swivel_vocab))

print("..loading swivel models", end="")
task_start_time = time.time()
swivel_model = {
    "given": load_swivel_model(given_swivel_model_path, len(swivel_vocab["given"])),
    "surname": load_swivel_model(surname_swivel_model_path, len(swivel_vocab["surname"])),
}
task_end_time = time.time()
print(f" {task_end_time - task_start_time} seconds")  # memory=", "unavailable")

print("..loading encoder models", end="")
task_start_time = time.time()
encoder_model = {
    "given": load_encoder_model(given_encoder_model_path),
    "surname": load_encoder_model(surname_encoder_model_path),
}
task_end_time = time.time()
print(f" {task_end_time - task_start_time} seconds")  # memory=", "unavailable")

print("..loading clusters", end="")
task_start_time = time.time()
clusters = {
    "given": load_clusters(given_clusters_path, swivel_vocab["given"], swivel_model["given"], encoder_model["given"]),
    "surname": load_clusters(
        surname_clusters_path, swivel_vocab["surname"], swivel_model["surname"], encoder_model["surname"]
    ),
}
task_end_time = time.time()
print(f" {task_end_time - task_start_time} seconds")  # memory=", get_size(clusters))

print("..loading nicknames", end="")
task_start_time = time.time()
name2variants = load_nicknames(nicknames_path)
task_end_time = time.time()
print(f" {task_end_time - task_start_time} seconds")  # memory=", get_size(clusters))

end_time = time.time()
print(f"Ready to serve in {end_time - start_time} seconds...", flush=True)
app = FastAPI()


class GivenSurname(str, Enum):
    given = "given"
    surname = "surname"


class NormalizeResponse(BaseModel):
    normalized: list[str]


class VectorResponse(BaseModel):
    vector: list[float]


class ClusterScore(BaseModel):
    id: str
    score: float

    @validator("score")
    def result_check(cls, v):
        return round(v, 3)


class StandardResponse(BaseModel):
    standards: list[ClusterScore]


@app.get("/normalize/{given_surname}/{name}", response_model=NormalizeResponse)
def normalize_api(given_surname: GivenSurname, name: str):
    """
    Normalize name
    """
    return NormalizeResponse(
        normalized=normalize(
            name, is_surname=given_surname == GivenSurname.surname, preserve_wildcards=False, handle_patronymics=False
        )
    )


@app.get("/vector/{given_surname}/{name}", response_model=VectorResponse)
def vector_api(given_surname: GivenSurname, name: str):
    """
    Name must be normalized, but this function standardizes patronymics in case they haven't already been standardized
    """
    if given_surname == GivenSurname.surname:
        name = standardize_patronymics(name)
    # get embedding
    name_embedding = get_swivel_embeddings(
        model=swivel_model[given_surname],
        vocab=swivel_vocab[given_surname],
        names=[add_padding(name)],
        encoder_model=encoder_model[given_surname],
    )[0]
    return VectorResponse(vector=name_embedding.tolist())


@app.get("/standard/{given_surname}/{name}", response_model=StandardResponse)
def standard_api(given_surname: GivenSurname, name: str):
    """
    Name must be normalized, but this function standardizes patronymics in case they haven't already been standardized
    """
    if given_surname == GivenSurname.surname:
        name = standardize_patronymics(name)
    name = add_padding(name)
    # get name variants if given names
    variants = list(name2variants.get(name, [name])) if given_surname == GivenSurname.given else [name]
    # get embedding
    variant_embeddings = get_swivel_embeddings(
        model=swivel_model[given_surname],
        vocab=swivel_vocab[given_surname],
        names=variants,
        encoder_model=encoder_model[given_surname],
    )
    # get clusters
    name2clusters, _ = get_clusters(
        all_names=variants,
        all_embeddings=variant_embeddings,
        clusters=clusters[given_surname].clustered_name_cluster_ids,
        cluster_embeddings=clusters[given_surname].clustered_name_embeddings,
        k=1024,
        max_clusters=max_search_clusters,
        verbose=False,
    )
    result = name2clusters[name] if len(name2clusters) == 1 else merge_name2clusters(name2clusters)
    result = [ClusterScore(id=cluster_score[0], score=cluster_score[1]) for cluster_score in result]
    return StandardResponse(standards=result)
