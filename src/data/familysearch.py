from collections import Counter

import pandas as pd

from src.data.utils import add_weighted_count, _add_padding


def train_test_split_on_frequency(
    dataset_path: str,
    train_path: str,
    test_path: str,
    threshold: float,
):
    """
    Split dataset_path into train and test based upon whether the name appears in pref tree names
    :param dataset_path: Path to input dataset
    :param train_path: Path to output training dataset
     - includes dataset instances where alt_name is above the train_cutoff in preferred tree names
    :param test_path: Path to output test dataset
     - includes dataset instances where alt_name is below the train_cutoff in preferred tree names
    :param threshold: percent of most-frequent names to put in training
    """
    # read the pairs
    print("read pairs")
    df = pd.read_csv(dataset_path)
    df.dropna(inplace=True)
    # drop unused columns and free memory
    keep_cols = ["name", "alt_name", "frequency"]
    df = df[keep_cols].copy()
    # change the schema to conform to the ancestry schema
    df.rename(columns={"name": "name1", "alt_name": "name2", "frequency": "co_occurrence"}, inplace=True)

    # remove illegal names (these can happen because of the loophole in normalize
    df = df[df["name1"].str.match("^[a-z]+$") & df["name2"].str.match("^[a-z]+$")]
    # remove single-character names
    df = df[(df["name1"].str.len() > 1) & (df["name2"].str.len() > 1)]

    # count overall name frequencies
    name_counter = Counter()
    for name1, name2, freq in zip(df["name1"], df["name2"], df["co_occurrence"]):
        name_counter[name1] += freq
        name_counter[name2] += freq

    # train names are the most-frequent threshold names
    k = int(len(name_counter) * threshold)
    train_names = set([name for name, _ in name_counter.most_common(k)])

    # train - when both tree and record names are frequent
    print("add weighted count to train")
    train_df = add_weighted_count(df[df["name1"].isin(train_names) & df["name2"].isin(train_names)].copy())

    # test - when either tree or record are rare (not frequent)
    print("add weighted count to test")
    test_df = add_weighted_count(df[~df["name1"].isin(train_names) | ~df["name2"].isin(train_names)].copy())

    # add padding
    print("add padding")
    train_df = _add_padding(train_df)
    test_df = _add_padding(test_df)

    # Persist splits on disk
    print("persist")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
