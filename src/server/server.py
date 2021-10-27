from fastapi import FastAPI, HTTPException
import jellyfish
import numpy as np
import pandas as pd
import torch

from src.models import utils

# run using uvicorn src.server.server:app --reload

# TODO load separate given and surname models
given_filename = "data/models/anc-triplet-bilstm-100-512-40-05.pth"
surname_filename = "data/models/anc-triplet-bilstm-100-512-40-05.pth"
given_clusters_filename = "data/models/surname_clusters.tsv"
surname_clusters_filename = "data/models/surname_clusters.tsv"

models = {
    "given": torch.load(given_filename),
    "surname": torch.load(surname_filename),
}
char_to_idx_map, idx_to_char_map = utils.build_token_idx_maps()
MAX_NAME_LENGTH = 30


def load_clusters(filename):
    df = pd.read_csv(filename, sep="\t", names=["id", "name"])
    df["name"] = df["name"].str.split(" ")
    return df.explode("name").reset_index(drop=True)


def calc_similarity_to(name):
    def calc_similarity(cand_name):
        dist = jellyfish.damerau_levenshtein_distance(name, cand_name)
        return 1 - (dist / max(len(name), len(cand_name)))

    return calc_similarity


def get_closest_id(name, candidate_names):
    scores = candidate_names["name"].apply(calc_similarity_to(name))
    closest_idx = np.argmax(scores)
    return candidate_names.iloc[closest_idx]["id"]


given_clusters_df = load_clusters(given_clusters_filename)
surname_clusters_df = load_clusters(surname_clusters_filename)

app = FastAPI()


@app.get("/vector/{given_surname}/{name}")
def vector(given_surname: str, name: str):
    if given_surname != "given" and given_surname != "surname":
        raise HTTPException(status_code=400, detail="path must be /vector/given or surname/name")
    # convert name to a tensor
    names_tensor, _ = utils.convert_names_to_model_inputs([name], char_to_idx_map, MAX_NAME_LENGTH)
    # Get Embeddings for the names from the encoder
    names_encoded = models[given_surname](names_tensor, just_encoder=True).detach().numpy()
    return {"vector": names_encoded[0].tolist()}


@app.get("/standard/{given_surname}/{name}")
def standard(given_surname: str, name: str):
    if given_surname != "given" and given_surname != "surname":
        raise HTTPException(status_code=400, detail="path must be /standard/given or surname/name")
    df = given_clusters_df if given_surname == "given" else surname_clusters_df
    standards = df[df["name"] == name]["id"].to_list()
    if len(standards) == 0:
        standards = [get_closest_id(name, df)]
    standards = [std.upper() for std in standards]
    return {"standard": standards}
