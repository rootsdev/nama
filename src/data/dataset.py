import numpy as np
import pandas as pd


def train_test_split(df: pd.DataFrame, target_label: str, test_size: float = 0.1) -> (pd.DataFrame, pd.DataFrame):
    """
    Split into train and test sets, with the condition that a target_label value cannot appear in both train and test splits
    """
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

    assert not len(set(df_train[target_label].tolist()).intersection(set(df_test[target_label].tolist())))

    return df_train, df_test
