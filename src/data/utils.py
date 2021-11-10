import numpy as np
import pandas as pd


def _train_test_split(
    df: pd.DataFrame, source_label: str, target_label: str, starting_test_size: float = 0.2
) -> (pd.DataFrame, pd.DataFrame):
    """
    Split into train and test sets,
    with the condition that a target_label value cannot appear in both train and test splits
    and the condition that a source_label value cannot appear in both train and test splits
    """

    # start with a test set twice what was asked for
    msk = np.random.uniform(0, 1, len(df)) < 1 - starting_test_size
    df_train = df[msk].copy()
    df_test = df[~msk].copy()

    # loop until no overlapping names
    while True:
        # Find target names that are both in train and test
        train_names = list(df_train[target_label].unique())
        msk = df_test[target_label].isin(train_names)
        df_duplicated = df_test[msk].copy()
        # Remove duplicated names from test and add it to train
        df_test.drop(df_test[msk].index, inplace=True)
        df_train = pd.concat([df_train, df_duplicated], axis=0)

        # Find source names that are both in train and test
        train_names = list(df_train[source_label].unique())
        msk = df_test[source_label].isin(train_names)
        df_duplicated = df_test[msk].copy()
        if df_duplicated.shape[0] == 0:
            break
        # Remove duplicated names from test and add it to train
        df_test.drop(df_test[msk].index, inplace=True)
        df_train = pd.concat([df_train, df_duplicated], axis=0)

    assert not len(set(df_train[target_label].tolist()).intersection(set(df_test[target_label].tolist())))
    assert not len(set(df_train[source_label].tolist()).intersection(set(df_test[source_label].tolist())))

    return df_train, df_test
