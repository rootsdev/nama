import pandas as pd

from src.data.utils import add_weighted_count, _add_padding


def train_test_split(
    pref_path: str,
    dataset_path: str,
    train_path: str,
    test_path: str,
    tiny_train_path: str,
    freq_train_path: str,
    tiny_test_path: str,
    freq_test_path: str,
    tiny_cutoff: int,
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
    :param tiny_train_path: Path to a dataset used during dev - very-frequent names to their in-vocabulary variants
     - in-vocabulary variant means alt_name is above the train_cutoff in preferred tree names
    :param freq_train_path: Path to a dataset used to test frequent names to their in-vocabulary variants
     - in-vocabulary variant means alt_name is above the train_cutoff in preferred tree names
    :param tiny_test_path: Path to a dataset used during dev - very-frequent names to their out-of-vocabulary variants
     - out-of-vocabulary variant means alt_name is below the train_cutoff in preferred tree names
    :param freq_test_path: Path to a dataset used to test frequent names to their out-of-vocabulary variants
     - out-of-vocabulary variant means alt_name is below the train_cutoff in preferred tree names
    :param tiny_cutoff: Names above this cutoff in preferred tree names will be used in the tiny datasets
    :param freq_cutoff: Names above this cutoff in preferred tree names are considered frequent
    :param train_cutoff: Alt-names above this cutoff in preferred tree names will appear in training
     - if 0 is passed in, use alt-names that appear anywhere in preferred tree names
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

    # read the preferred names
    print("read preferred names")
    pref_df = pd.read_csv(pref_path)

    # get tiny, frequent, and training names
    print("calc train and test splits")
    tiny_names = set(pref_df["name"][:tiny_cutoff])
    freq_names = set(pref_df["name"][:freq_cutoff])
    train_names = set(pref_df["name"] if train_cutoff == 0 else pref_df["name"][:train_cutoff])

    # train
    print("add weighted count to train")
    train_df = add_weighted_count(df[df["name2"].isin(train_names)].copy())

    # test
    print("add weighted count to test")
    test_df = add_weighted_count(df[~df["name2"].isin(train_names)].copy())

    # tiny train (for testing in-vocab variants during development)
    print("add weighted count to tiny_train")
    tiny_train_df = add_weighted_count(train_df[train_df["name1"].isin(tiny_names)].copy())

    # freq train (for testing in-vocab variants)
    print("add weighted count to freq_train")
    freq_train_df = add_weighted_count(train_df[train_df["name1"].isin(freq_names)].copy())

    # tiny test (for testing out-of-vocab variants)
    print("add weighted count to tiny_test")
    tiny_test_df = add_weighted_count(test_df[test_df["name1"].isin(tiny_names)].copy())

    # freq test (for testing out-of-vocab variants)
    print("add weighted count to freq_test")
    freq_test_df = add_weighted_count(test_df[test_df["name1"].isin(freq_names)].copy())

    # add padding
    print("add padding")
    train_df = _add_padding(train_df)
    test_df = _add_padding(test_df)
    tiny_train_df = _add_padding(tiny_train_df)
    freq_train_df = _add_padding(freq_train_df)
    tiny_test_df = _add_padding(tiny_test_df)
    freq_test_df = _add_padding(freq_test_df)

    # Persist splits on disk
    print("persist")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    tiny_train_df.to_csv(tiny_train_path, index=False)
    freq_train_df.to_csv(freq_train_path, index=False)
    tiny_test_df.to_csv(tiny_test_path, index=False)
    freq_test_df.to_csv(freq_test_path, index=False)
