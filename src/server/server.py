from enum import Enum
from fastapi import FastAPI, HTTPException
import jellyfish
import numpy as np
import pandas as pd
from pydantic import BaseModel
import torch
from typing import List

from src.data.prepare import standardize_patronymics
from models.autoencoder import convert_names_to_model_inputs

# run using uvicorn src.server.server:app --reload

# TODO load separate given and surname models
given_filename = "data/models/anc-triplet-bilstm-100-512-40-05.pth"
surname_filename = "data/models/anc-triplet-bilstm-100-512-40-05.pth"
given_clusters_filename = "data/models/surname_clusters.tsv"
surname_clusters_filename = "data/models/surname_clusters.tsv"


def load_clusters(filename):
    df = pd.read_csv(filename, sep="\t", names=["id", "name"])
    df["name"] = df["name"].str.split(" ")
    return df.explode("name").reset_index(drop=True)


models = {
    "given": torch.load(given_filename),
    "surname": torch.load(surname_filename),
}
clusters_df = {
    "given": load_clusters(given_clusters_filename),
    "surname": load_clusters(surname_clusters_filename),
}


class GivenSurname(str, Enum):
    given = "given"
    surname = "surname"


class VectorResponse(BaseModel):
    vector: List[float]


class StandardResponse(BaseModel):
    standard: List[str]


def calc_similarity_to(name):
    def calc_similarity(cand_name):
        dist = jellyfish.damerau_levenshtein_distance(name, cand_name)
        return 1 - (dist / max(len(name), len(cand_name)))

    return calc_similarity


def get_closest_id(name, candidate_names):
    scores = candidate_names["name"].apply(calc_similarity_to(name))
    closest_idx = np.argmax(scores)
    return candidate_names.iloc[closest_idx]["id"]


app = FastAPI()


@app.get("/vector/{given_surname}/{name}", response_model=VectorResponse)
def vector(given_surname: GivenSurname, name: str):
    """
    Name must be normalized, but this function standardizes patronymics in case they haven't already been standardized
    """
    if given_surname == GivenSurname.surname:
        name = standardize_patronymics(name)
    # convert name to a tensor
    names_X, _ = convert_names_to_model_inputs([name])
    # Get embeddings for the name from the encoder
    name_embeddings = models[given_surname](names_X, just_encoder=True).detach().numpy()
    return VectorResponse(vector=name_embeddings[0].tolist())


@app.get("/standard/{given_surname}/{name}", response_model=StandardResponse)
def standard(given_surname: GivenSurname, name: str):
    """
    Name must be normalized, but this function standardizes patronymics in case they haven't already been standardized
    """
    if given_surname == GivenSurname.surname:
        name = standardize_patronymics(name)
    df = clusters_df[given_surname]
    standards = df[df["name"] == name]["id"].to_list()
    if len(standards) == 0:
        standards = [get_closest_id(name, df)]
    standards = [std.upper() for std in standards]
    return StandardResponse(standard=standards)
