from collections import Counter
from typing import List, Tuple, Set

import numpy as np
import pandas as pd
from sklearn import model_selection

from src.models.utils import add_padding


def load_datasets(paths: List[str]) -> List[Tuple[List[str], List[List[Tuple[str, float, int]]], np.array]]:
    return [_load(pd.read_csv(path)) for path in paths]


def filter_dataset(input_names: List[str],
                    weighted_actual_names: List[List[Tuple[str, float, int]]],
                    selected_names: Set[str]
                    ) -> Tuple[List[str], List[List[Tuple[str, float, int]]], np.array]:
    """
    Filter dataset to have only selected_names, reweighting actual names
    """
    input_names_filtered = []
    weighted_actual_names_filtered = []
    candidate_names_filtered = set()
    for input_name, wans in zip(input_names, weighted_actual_names):
        wans = [(name, 0.0, freq) for name, _, freq in wans if input_name in selected_names and name in selected_names]
        if len(wans) > 0:
            # re-weight
            total_freq = sum([freq for _, _, freq in wans])
            if total_freq > 0:  # if total_freq is 0, then the only candidate_name is the input name, so we can skip
                wans = [(name, freq / total_freq, freq) for name, _, freq in wans]
                input_names_filtered.append(input_name)
                weighted_actual_names_filtered.append(wans)
                candidate_names_filtered.update([name for name, _, _ in wans])
    candidate_names_filtered = np.array([name for name in candidate_names_filtered])
    return input_names_filtered, weighted_actual_names_filtered, candidate_names_filtered


def train_test_split(input_names: List[str],
                     weighted_actual_names: List[List[Tuple[str, float, int]]],
                     candidate_names: np.array,
                     train_size=None,
                     test_size=None,
                     ) -> List[Tuple[List[str], List[List[Tuple[str, float, int]]], np.array]]:
    """
    Split input_names, weighted_actual_names, and candidate names into two subsets:
    one where the input names and candidate names are both in the train subset, and
    one where the input names and candidate names are both in the test subset set
    """
    all_names = list(set(input_names).union(set(candidate_names)))
    train_names, test_names = model_selection.train_test_split(all_names, train_size=train_size, test_size=test_size)
    train_names = set(train_names)
    test_names = set(test_names)

    input_names_train, weighted_actual_names_train, candidate_names_train = \
        filter_dataset(input_names, weighted_actual_names, train_names)
    input_names_test, weighted_actual_names_test, candidate_names_test = \
        filter_dataset(input_names, weighted_actual_names, test_names)
    return [(input_names_train, weighted_actual_names_train, candidate_names_train),
            (input_names_test, weighted_actual_names_test, candidate_names_test)]


def frequent_k_names(input_names, weighted_actual_names, k, input_names_only=False):
    name_counter = Counter()
    for input_name, wan in zip(input_names, weighted_actual_names):
        for name, _, co_occurrence in wan:
            name_counter[input_name] += co_occurrence
            if not input_names_only:
                name_counter[name] += co_occurrence
    return [name for name, _ in name_counter.most_common(k)]


def select_frequent_k(input_names: List[str],
                      weighted_actual_names: List[List[Tuple[str, float, int]]],
                      candidate_names: np.array,
                      k,
                      input_names_only=False,
                      ) -> Tuple[List[str], List[List[Tuple[str, float, int]]], np.array]:
    """
    Filter dataset to have only the most-frequent k names
    """
    selected_names = set(frequent_k_names(input_names, weighted_actual_names, k, input_names_only))

    return filter_dataset(input_names, weighted_actual_names, selected_names)


def add_weighted_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, calculate the correct weighted_count and add padding
    """

    def divide_weighted_count_by_sum(df):
        df["weighted_count"] /= df["weighted_count"].sum()
        return df

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
    return df


def _add_padding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add padding to name1 and name2
    :param df: input dataframe
    :return: input dataframe with name1 and name2 columns padded
    """
    df.loc[:, "name1"] = df.loc[:, "name1"].map(add_padding)
    df.loc[:, "name2"] = df.loc[:, "name2"].map(add_padding)
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

    # add (name1, 0.0, 0) to each weighted_actual_names list
    # so if a name matches itself, it doesn't hurt precision
    for ix in range(0, len(input_names)):
        name1 = input_names[ix]
        weighted_actual_names[ix].append((name1, 0.0, 0))

    # if you want just relevant names:
    # [[name for name,weight in name_weights] for name_weights in weighted_actual_names]
    return input_names, weighted_actual_names, candidate_names
