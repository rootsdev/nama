from collections import Counter, defaultdict
import csv
import sys
from typing import List, Tuple, Set

import numpy as np
import pandas as pd
from sklearn import model_selection

from src.data.filesystem import fopen
from src.models.utils import add_padding


def read_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV into a dataframe
    """
    return pd.read_csv(path, na_filter=False)


def load_dataset(path: List[str], is_eval=False, verbose=False) -> Tuple[List[str], List[List[Tuple[str, float, int]]], np.array]:
    """
    load name1, name2, weight, co-occurrence rows from a CSV and return
    a list of input names (distinct name1),
    a list of lists of weighted actual names (name2, weighted_count (probability that name1 -> name2, and co_occurrence)
     for each input name
    a list of candidate names (distinct name2)
    is_eval: if True, remove any names associated with themselves and re-calculate the weights
    because we don't want to have a name matching itself increase the weighted recall
    """
    # read dataframe
    if verbose:
        print("Reading dataset from {}".format(path))
    df = pd.read_csv(path)
    # if we're using this dataset for evaluation, remove self-matches
    # so matching the same name doesn't increase recall
    if is_eval:
        df.drop(df[df["name1"] == df["name2"]].index, inplace=True)
        # and re-weight
        if verbose:
            print("Re-weighting dataset")
        df = add_weighted_count(df)

    if verbose:
        print("Calculating weighted actual names")
    candidate_names = np.array(df["name2"].unique())
    df_name_matches = df.groupby("name1").agg(list).reset_index()
    df = None
    input_names = df_name_matches["name1"].tolist()
    weighted_actual_names = [
        [(n, w, c) for n, w, c in zip(ns, ws, cs)]
        for ns, ws, cs in zip(
            df_name_matches["name2"], df_name_matches["weighted_count"], df_name_matches["co_occurrence"]
        )
    ]
    df_name_matches = None

    # add (name1, 0.0, 0) to each weighted_actual_names list if it doesn't already exist
    # so if a name matches itself, it doesn't hurt precision
    if verbose:
        print("Adding padding to weighted actual names")
    for ix in range(0, len(input_names)):
        name1 = input_names[ix]
        if not any(name == name1 for name, _, _ in weighted_actual_names[ix]):
            weighted_actual_names[ix].append((name1, 0.0, 0))

    # if you want just relevant names:
    # [[name for name,weight in name_weights] for name_weights in weighted_actual_names]
    return input_names, weighted_actual_names, candidate_names


def load_dataset_v2(path: List[str], verbose=False) -> Tuple[List[str], List[List[Tuple[str, int]]], List[str]]:
    """
    load tree_name, record_name, frequency rows from a CSV and return
    a list of tree names (distinct tree_name),
    a list of lists of attached record names (record_name and frequency) for each input name
    a list of record names (distinct record_name)
    because we don't want to have a name matching itself increase the weighted recall
    """
    # read dataframe
    if verbose:
        print("Reading dataset from {}".format(path))
    df = read_csv(path)
    # if we're using this dataset for evaluation, remove self-matches
    # so matching the same name doesn't increase recall

    if verbose:
        print("Calculating weighted actual names")
    record_names = df["record_name"].unique().tolist()
    df_name_matches = df.groupby("tree_name").agg(list).reset_index()
    df = None
    tree_names = df_name_matches["tree_name"].tolist()
    attached_names = [
        [(record_name, frequency) for record_name, frequency in zip(record_names, frequencies)]
        for record_names, frequencies in zip(df_name_matches["record_name"], df_name_matches["frequency"])
    ]
    df_name_matches = None

    # add (name1, 0) to each weighted_actual_names list if it doesn't already exist
    # so if a name matches itself, it doesn't hurt precision
    if verbose:
        print("Adding padding to weighted actual names")
    for ix in range(0, len(tree_names)):
        name1 = tree_names[ix]
        if not any(name == name1 for name, _ in attached_names[ix]):
            attached_names[ix].append((name1, 0))

    return tree_names, attached_names, record_names


def filter_dataset(input_names: List[str],
                    weighted_actual_names: List[List[Tuple[str, float, int]]],
                    selected_names: Set[str],
                    all_actuals=False,
                    ) -> Tuple[List[str], List[List[Tuple[str, float, int]]], np.array]:
    """
    Filter dataset to have only selected_names, reweighting actual names
    """
    input_names_filtered = []
    weighted_actual_names_filtered = []
    candidate_names_filtered = set()
    for input_name, wans in zip(input_names, weighted_actual_names):
        if input_name not in selected_names:
            continue
        if not all_actuals:
            wans = [(name, 0.0, freq) for name, _, freq in wans if input_name in selected_names and name in selected_names]
            # re-weight
            total_freq = sum([freq for _, _, freq in wans])
            if total_freq == 0:  # if total_freq is 0, then the only candidate_name is the input name, so we can skip
                continue
            wans = [(name, freq / total_freq, freq) for name, _, freq in wans]
        input_names_filtered.append(input_name)
        weighted_actual_names_filtered.append(wans)
        candidate_names_filtered.update([name for name, _, _ in wans])
    # may be faster to say np.array(list(candidate_names_filtered))
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
                      all_actuals=False,
                      ) -> Tuple[List[str], List[List[Tuple[str, float, int]]], np.array]:
    """
    Filter dataset to have only the most-frequent k names
    """
    selected_names = set(frequent_k_names(input_names, weighted_actual_names, k, input_names_only))

    return filter_dataset(input_names, weighted_actual_names, selected_names, all_actuals=all_actuals)


def add_weighted_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, add the correct weighted_count
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


def load_nicknames(path):
    name2variants = defaultdict(set)
    with fopen(path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            for name1 in line:
                name1 = add_padding(name1)
                for name2 in line:
                    name2 = add_padding(name2)
                    name2variants[name1].add(name2)
    return name2variants


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size