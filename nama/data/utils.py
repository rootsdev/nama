import csv
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn import model_selection

from nama.models.utils import add_padding


def read_csv(path: str) -> pd.DataFrame:
    """
    Read a CSV into a dataframe
    """
    return pd.read_csv(path, na_filter=False)


def load_dataset(path: list[str], verbose=False) -> tuple[list[str], list[list[tuple[str, int]]], list[str]]:
    """
    load tree_name, record_name, frequency rows from a CSV and return
    a list of tree names (distinct tree_name),
    a list of lists of attached record names (record_name and frequency) for each input name
    a list of record names (distinct record_name)
    because we don't want to have a name matching itself increase the weighted recall
    """
    # read dataframe
    if verbose:
        print(f"Reading dataset from {path}")
    df = read_csv(path)
    # if we're using this dataset for evaluation, remove self-matches
    # so matching the same name doesn't increase recall

    if verbose:
        print("Calculating record name frequencies")
    record_names = df["record_name"].unique().tolist()
    df_name_matches = df.groupby("tree_name").agg(list).reset_index()
    df = None
    tree_names = df_name_matches["tree_name"].tolist()
    attached_names = [
        list(zip(record_names, frequencies))
        for record_names, frequencies in zip(df_name_matches["record_name"], df_name_matches["frequency"])
    ]
    df_name_matches = None

    # add (name1, 0) to each record_name_frequencies list if it doesn't already exist
    # so if a name matches itself, it doesn't hurt precision
    if verbose:
        print("Adding padding to record name frequencies")
    for ix in range(0, len(tree_names)):
        name1 = tree_names[ix]
        if not any(name == name1 for name, _ in attached_names[ix]):
            attached_names[ix].append((name1, 0))

    return tree_names, attached_names, record_names


def filter_dataset(
    input_names: list[str],
    record_name_frequencies: list[list[tuple[str, float, int]]],
    selected_names: set[str],
    all_record_name_frequencies=False,
) -> tuple[list[str], list[list[tuple[str, float, int]]], np.array]:
    """
    Filter dataset to have only selected_names
    """
    input_names_filtered = []
    record_name_frequencies_filtered = []
    candidate_names_filtered = set()
    for input_name, rnfs in zip(input_names, record_name_frequencies):
        if input_name not in selected_names:
            continue
        if not all_record_name_frequencies:
            rnfs = [(name, freq) for name, freq in rnfs if input_name in selected_names and name in selected_names]
        input_names_filtered.append(input_name)
        record_name_frequencies_filtered.append(rnfs)
        candidate_names_filtered.update([name for name, _ in rnfs])
    # may be faster to say np.array(list(candidate_names_filtered))
    candidate_names_filtered = np.array(list(candidate_names_filtered))
    return input_names_filtered, record_name_frequencies_filtered, candidate_names_filtered


def train_test_split(
    input_names: list[str],
    record_name_frequencies: list[list[tuple[str, float, int]]],
    candidate_names: np.array,
    train_size=None,
    test_size=None,
) -> list[tuple[list[str], list[list[tuple[str, float, int]]], np.array]]:
    """
    Split input_names, record_name_frequencies, and candidate names into two subsets:
    one where the input names and candidate names are both in the train subset, and
    one where the input names and candidate names are both in the test subset set
    """
    all_names = list(set(input_names).union(set(candidate_names)))
    train_names, test_names = model_selection.train_test_split(all_names, train_size=train_size, test_size=test_size)
    train_names = set(train_names)
    test_names = set(test_names)

    input_names_train, record_name_frequencies_train, candidate_names_train = filter_dataset(
        input_names, record_name_frequencies, train_names
    )
    input_names_test, record_name_frequencies_test, candidate_names_test = filter_dataset(
        input_names, record_name_frequencies, test_names
    )
    return [
        (input_names_train, record_name_frequencies_train, candidate_names_train),
        (input_names_test, record_name_frequencies_test, candidate_names_test),
    ]


def frequent_k_names(input_names, record_name_frequencies, k, input_names_only=False):
    name_counter = Counter()
    for input_name, rnf in zip(input_names, record_name_frequencies):
        for name, frequency in rnf:
            name_counter[input_name] += frequency
            if not input_names_only:
                name_counter[name] += frequency
    return [name for name, _ in name_counter.most_common(k)]


def select_frequent_k(
    input_names: list[str],
    record_name_frequencies: list[list[tuple[str, float, int]]],
    candidate_names: np.array,
    k,
    input_names_only=False,
    all_record_name_frequencies=False,
) -> tuple[list[str], list[list[tuple[str, float, int]]], np.array]:
    """
    Filter dataset to have only the most-frequent k names
    """
    selected_names = set(frequent_k_names(input_names, record_name_frequencies, k, input_names_only))

    return filter_dataset(
        input_names, record_name_frequencies, selected_names, all_record_name_frequencies=all_record_name_frequencies
    )


def _add_padding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add padding to tree_name and record_name
    :param df: input dataframe
    :return: input dataframe with tree_name and record_name columns padded
    """
    df.loc[:, "tree_name"] = df.loc[:, "tree_name"].map(add_padding)
    df.loc[:, "record_name"] = df.loc[:, "record_name"].map(add_padding)
    return df


def load_nicknames(path):
    name2variants = defaultdict(set)
    with open(path) as f:
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
        size += sum([get_size(k, seen) for k in obj])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
