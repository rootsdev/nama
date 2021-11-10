import pandas as pd

from src.data.utils import _prepare


def train_test_split(
    pref_path: str,
    dataset_path: str,
    train_path: str,
    test_path: str,
    freq_train_path: str,
    freq_test_path: str,
    freq_cutoff: int,
    train_cutoff: int = 0,
):
    """
    Split dataset_path into four files based upon freq_cutoff and train_cutoff
    :param pref_path: Path to a list of preferred tree names - used to identify frequent names for cutoffs
    :param dataset_path: Path to input dataset
    :param train_path: Path to output training dataset
     - includes dataset instances where alt_name is above the train_cutoff in preferred tree names
    :param test_path: Path to output test dataset
     - includes dataset instances where alt_name is below the train_cutoff in preferred tree names
    :param freq_train_path: Path to a dataset used to test frequent names to their in-vocabulary variants
     - in-vocaulary variant means alt_name is above the train_cutoff in preferred tree names
    :param freq_test_path: Path to a dataset used to test frequent names to their out-of-vocabulary variants
     - out-of-vocaulary variant means alt_name is below the train_cutoff in preferred tree names
    :param freq_cutoff: Names above this cutoff in preferred tree names are considered frequent
    :param train_cutoff: Alt-names above this cutoff in preferred tree names will appear in training
     - if 0 is passed in, use alt-names that appear anywhere in preferred tree names
    """
    # read the pairs
    df = pd.read_csv(dataset_path)
    df.dropna(inplace=True)
    # drop unused columns and free memory
    keep_cols = ["name", "alt_name", "frequency"]
    df = df[keep_cols].copy()
    # change the schema to conform to the ancestry schema
    df.rename(columns={"name": "name1", "alt_name": "name2", "frequency": "co_occurrence"}, inplace=True)

    # read the preferred names
    pref_df = pd.read_csv(pref_path)

    # get frequent names and training names
    freq_names = set(pref_df["name"][:freq_cutoff])
    train_names = set(pref_df["name"] if train_cutoff == 0 else pref_df["name"][:train_cutoff])

    # train
    train_df = _prepare(df[df["name2"].isin(train_names)].copy())

    # test
    test_df = _prepare(df[~df["name2"].isin(train_names)].copy())

    # freq train (for testing in-vocab variants)
    freq_train_df = _prepare(train_df[train_df["name1"].isin(freq_names)].copy())

    # freq test (for testing out-of-vocab variants)
    freq_test_df = _prepare(test_df[test_df["name1"].isin(freq_names)].copy())

    # Persist splits on disk
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)
    freq_train_df.to_csv(freq_train_path)
    freq_test_df.to_csv(freq_test_path)
