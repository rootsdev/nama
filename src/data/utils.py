import numpy as np
import pandas as pd
from typing import List, Tuple

from src.models.utils import add_padding


def load_train_test(paths: List[str]) -> List[Tuple[List[str], List[List[Tuple[str, float, int]]], np.array]]:
    """
    Load and process train and test datasets
    """
    return [_load(pd.read_csv(path)) for path in paths]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, calculate the correct weighted_count
    """

    def divide_weighted_count_by_sum(df):
        df["weighted_count"] /= df["weighted_count"].sum()
        return df

    # Add padding
    df.loc[:, "name1"] = df.loc[:, "name1"].map(add_padding)
    df.loc[:, "name2"] = df.loc[:, "name2"].map(add_padding)
    # weighted_count will be the co-occurrence / sum(co-occurrences)
    df.loc[:, "weighted_count"] = df.loc[:, "co_occurrence"]

    # probably doesn't have to group the first time for familysearch,
    # but it doesn't hurt and it may be required for ancestry
    df = (
        df.groupby(["name1", "name2"])
        .agg({"weighted_count": "sum", "co_occurrence": "sum"})
        .groupby(level=0)
        .apply(divide_weighted_count_by_sum)
        .reset_index()
    )
    # TODO remove co_occurrence count if we don't use it
    return df


def _load(df: pd.DataFrame) -> (List[str], List[List[Tuple[str, float, int]]], np.array):
    """
    Given a dataframe, return
    a list of input names (distinct name1),
    a list of lists of weighted actual names (name2, weighted_count (probability that name1 -> name2, and co_occurrence)
     for each input name
    a list of candidate names (distinct name2)
    """
    df_name_matches = df.groupby("name1").agg(list).reset_index()
    weighted_actual_names = [
        [(n, w, c) for n, w, c in zip(ns, ws, cs)]
        for ns, ws, cs in zip(
            df_name_matches["name2"], df_name_matches["weighted_count"], df_name_matches["co_occurrence"]
        )
    ]
    input_names = df_name_matches["name1"].tolist()
    candidate_names = np.array(df["name2"].unique())

    # if you want just relevant names:
    # [[name for name,weight in name_weights] for name_weights in weighted_actual_names]
    return input_names, weighted_actual_names, candidate_names
