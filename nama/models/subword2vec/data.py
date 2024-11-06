import numpy as np
import pandas as pd


def process_train_data(train_data_path: str, save_path: str, min_frequency: int = 10) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(train_data_path)
    df = df[df["frequency"] > min_frequency]

    # Group all alt names for each name
    df_names = df.groupby("name")["alt_name"].agg(list).reset_index()
    df_names.rename(columns={"alt_name": "alt_names"}, inplace=True)

    # Combine name and alt names into a single list
    df_names["all_names"] = df_names[["name", "alt_names"]].apply(lambda x: [x[0]] + x[1], axis=1)

    # Store maximum length of name + alt_names list
    # We need this to set the window size for the fastText model
    max_len = df_names["all_names"].str.len().max()

    # Convert list into a space separated string (fastText format for sequence)
    df_names["all_names"] = df_names["all_names"].map(lambda x: " ".join(x))

    # Save names as txt file to save_path
    df_names["all_names"].to_csv(save_path, encoding="utf-8", index=False, header=None)

    return df_names, max_len


def process_test_data(test_data_path: str):
    df_test = pd.read_csv(test_data_path)

    # Group all alt names for each name
    df_test_names = df_test.groupby("name")["alt_name"].agg(list).reset_index()
    df_test_names.rename(columns={"alt_name": "alt_names"}, inplace=True)

    # Store all names that occur in the test data as a flat list
    all_names = df_test_names[["name", "alt_names"]].apply(lambda x: [x[0]] + x[1], axis=1).tolist()
    all_names_flat = np.array(list({item for sublist in all_names for item in sublist}))

    return df_test_names, all_names_flat
