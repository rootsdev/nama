import numpy as np
import pandas as pd
from src.data.utils import _add_padding, add_weighted_count


def train_test_split(dataset_path: str, train_path: str, test_path: str, test_size: float = 0.1):
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
    target_label = "name2"
    msk = np.random.uniform(0, 1, len(df)) < 1 - test_size
    df_train = df[msk].copy()
    df_test = df[~msk].copy()

    # Find record names that are both in train and test
    train_names = list(df_train[target_label].unique())
    msk = df_test[target_label].isin(train_names)
    df_duplicated = df_test[msk].copy()

    # Remove duplicated names from test and add it to train
    df_test.drop(df_test[msk].index, inplace=True)
    df_train = pd.concat([df_train, df_duplicated], axis=0)

    # assert not len(set(df_train[target_label].tolist()).intersection(set(df_test[target_label].tolist())))

    # set the ordered_prob column and limit the columns to save
    df_train = add_weighted_count(df_train)
    df_train = _add_padding(df_train)
    df_test = add_weighted_count(df_test)
    df_test = _add_padding(df_test)

    # Persist splits on disk
    df_train.to_csv(train_path)
    df_test.to_csv(test_path)
