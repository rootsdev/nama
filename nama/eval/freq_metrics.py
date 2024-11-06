import pandas as pd
from tqdm import tqdm


def calc_precision_recall(query: str, cluster_names: set[str], df: pd.DataFrame) -> tuple[float, float]:
    # compute total frequency of query name
    query_freq = df[df["tree_name"] == query]["frequency"].sum()

    # compute total frequency of all cluster_names
    cluster_df = df[df["record_name"].isin(cluster_names)]
    cluster_freq = cluster_df["frequency"].sum()

    # compute precision = # of times cluster name is associated with query name / # of times cluster name appears
    cluster_query_freq = cluster_df[cluster_df["tree_name"] == query]["frequency"].sum()
    precision = cluster_query_freq / cluster_freq if cluster_freq > 0 else 1.0

    # compute recall = # times cluster name is associated with query name / # of times query name appears
    recall = cluster_query_freq / query_freq if query_freq > 0 else 1.0

    return precision, recall


def calc_avg_precision_recall(
    query_names: list[str], name2codes: dict[str, set[str]], code2names: dict[str, set[str]], df: pd.DataFrame
) -> tuple[float, float, float, float]:
    # exclude frequencies where record name == tree name
    # df = df[df["record_name"] != df["tree_name"]]

    sum_precision = 0.0
    sum_recall = 0.0
    for name in tqdm(query_names, mininterval=2.0):
        cluster_names = set().union(*[code2names[code] for code in name2codes[name]])
        precision, recall = calc_precision_recall(name, cluster_names, df)
        sum_precision += precision
        sum_recall += recall

    avg_precision = sum_precision / len(query_names)
    avg_recall = sum_recall / len(query_names)
    f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)
    f2 = (5 * avg_precision * avg_recall) / (4 * avg_precision + avg_recall)
    return avg_precision, avg_recall, f1, f2
