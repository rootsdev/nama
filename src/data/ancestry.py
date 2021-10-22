import pandas as pd
import numpy as np
from typing import List, Tuple

from data.dataset import train_test_split
from src.models import utils


def ancestry_train_test_split(dataset_path: str, train_path: str, test_path: str, test_size: float = 0.1):
    """
    Split test and train data in DATASET_NAME and save DATASET_NAME_train and DATASET_NAME_test to disk
    """
    df = pd.read_csv(dataset_path, sep="\t", header=None)
    # name1 = tree name
    # name2 = record name
    # co_occurrence = number of times name1 -> name2
    # count1 = total count of name1 (I think?) - not used
    # count2 = total count of name2 (I think?) - not used
    df.columns = ["name1", "name2", "co_occurrence", "count1", "count2"]
    df.dropna(inplace=True)

    # Split train test
    df_train, df_test = train_test_split(df, "name2", test_size=test_size)

    # Persist splits on disk
    df_train.to_csv(train_path)
    df_test.to_csv(test_path)


def load_ancestry_train_test(
    train_path: str, test_path: str
) -> (
    (List[str], List[List[Tuple[str, float, int]]], np.array),
    (List[str], List[List[Tuple[str, float, int]]], np.array),
):
    """
    Load and process test and train datasets
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    input_names_train, weighted_actual_names_train, all_candidates_train = process_ancestry(df_train)
    input_names_test, weighted_actual_names_test, all_candidates_test = process_ancestry(df_test)

    return (input_names_train, weighted_actual_names_train, all_candidates_train), (
        input_names_test,
        weighted_actual_names_test,
        all_candidates_test,
    )


def process_ancestry(df: pd.DataFrame) -> (List[str], List[List[Tuple[str, float, int]]], np.array):
    """
    Given a dataframe, return
    a list of input names (distinct name1),
    a list of lists of weighted actual names (name2, weighted_count (probability that name1 -> name2, and co_occurrence) for each input name
    a list of candidate names (distinct name2)
    """
    # TODO remove co_occurrence count if we don't use it
    def divide_weighted_count_by_sum(df):
        df["weighted_count"] /= df["weighted_count"].sum()
        return df

    # Add padding
    df.loc[:, "name1"] = df.loc[:, "name1"].map(utils.add_padding)
    df.loc[:, "name2"] = df.loc[:, "name2"].map(utils.add_padding)
    # weighted_count will be the co-occurrence / sum(co-occurrences)
    df.loc[:, "weighted_count"] = df.loc[:, "co_occurrence"]

    df = (
        df.groupby(["name1", "name2"])
        .agg({"weighted_count": "sum", "co_occurrence": "sum"})
        .groupby(level=0)
        .apply(divide_weighted_count_by_sum)
        .reset_index()
    )

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
